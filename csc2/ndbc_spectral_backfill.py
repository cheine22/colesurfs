"""CSC2 — NDBC spectral (swden + swdir) historical backfill.

Why this exists. The realtime obs logger (`csc2.obs_logger`) writes
spectrally-decomposed swell partitions (partition=1 primary, partition=2
secondary) by calling `buoy.fetch_buoy`, which pulls `.data_spec` and
`.swdir` from the realtime2 endpoint. That gives us partition=1/2 obs
*from the date the logger started forward only*. The historical
`.csc_data/observations/buoy=<id>/year=Y/stdmet*.parquet` archive contains
**partition=0 only** (combined-sea WVHT/DPD/MWD), because the stdmet
backfill never touched the spectral side.

Without partition=1/2 obs going back, the trainable target horizon
collapses to the obs_logger uptime (currently ~5 days), even though
EURO/GFS forecast cycles are paired against buoy obs across ~360 days.
This module closes that gap by re-running NDBC's published yearly /
monthly spectral archives through the **identical** decomposition the
dashboard uses (`buoy._parse_spectral_file_all_rows` +
`buoy._spectral_components`) and writing partition=1/2 rows alongside the
existing partition=0 stdmet shards.

Archive layout used (NDBC public free archives):
  - Yearly closed (~Q1 of following year):
      view_text_file.php?filename=<sta>w<year>.txt.gz&dir=data/historical/swden/
      view_text_file.php?filename=<sta>d<year>.txt.gz&dir=data/historical/swdir/
  - Monthly archived (closed, weeks after month-end):
      view_text_file.php?filename=<sta><M><year>.txt.gz&dir=data/swden/<Mon>/
      view_text_file.php?filename=<sta><M><year>.txt.gz&dir=data/swdir/<Mon>/
  - Monthly rolling (most-recent finalized month, uncompressed):
      data/swden/<Mon>/<sta>.txt
      data/swdir/<Mon>/<sta>.txt
  - The current month is intentionally NOT covered here — the live
    obs_logger covers it from realtime2.

File format note. The historical archive files use a *frequency-in-
header* layout (freqs listed in the comment line, then one row per
timestamp with values only). The realtime2 `.data_spec`/`.swdir` files
use *frequency-in-parens* (each value followed by `(freq)`). Same
underlying data, different parser. This module ships its own header-
format parser; the resulting `[(freq, value), ...]` shape is fed to the
dashboard's `_spectral_components` unchanged so the decomposition is
byte-identical to the live path.

Output. One parquet per (buoy, year) (yearly source) or (buoy, year,
month) (monthly source) at:
    .csc_data/observations/buoy=<id>/year=Y/spectral.parquet
    .csc_data/observations/buoy=<id>/year=Y/spectral-YYYY-MM.parquet
Each shard contains 0-2 rows per spectral observation timestamp
(partition=1 primary swell, partition=2 secondary swell — only included
when they clear the dashboard's Hm0 ≥ 0.2 ft / Tm ≥ 6 s gates). Schema
matches the rest of `.csc_data/observations/`:
    buoy_id, valid_utc, partition, hs_m, tp_s, dp_deg, source, ingest_utc

Usage:
    python -m csc2.ndbc_spectral_backfill                           # all east+west, 2021..today
    python -m csc2.ndbc_spectral_backfill --start-year 2021 --end-year 2025
    python -m csc2.ndbc_spectral_backfill --buoy 44065 --year 2024
    python -m csc2.ndbc_spectral_backfill --monthly-range 2026-01:2026-04
    python -m csc2.ndbc_spectral_backfill --force                   # overwrite existing
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone as dtz
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csc2.schema import BUOYS, LOGS_DIR, ensure_dirs  # noqa: E402
from buoy import (  # noqa: E402  — dashboard-identical decomposition
    _spectral_components,
    _parse_spectral_file_all_rows,
)

HIST_OBS_DIR = _ROOT / ".csc_data" / "observations"
HEADERS = {"User-Agent": "ColeSurfs/CSC2 spectral-backfill"}
FT_PER_M = 3.28084

# NDBC fill markers seen across spectral archives. 999.0 is the canonical
# direction fill; 99.0 surfaces in legacy density rows. Treat any of these
# as "missing this bin" — the row may still be salvageable if other bins
# are valid.
_FILL = {99.0, 99, 999.0, 999, 9999.0, 9999, -99.0, -999.0}

_MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def _yearly_url(station: str, year: int, kind: str) -> str:
    """kind ∈ {'swden', 'swdir'}. swden uses 'w', swdir uses 'd' as the
    NDBC filename suffix for yearly archives."""
    suffix = {"swden": "w", "swdir": "d"}[kind]
    return (f"https://www.ndbc.noaa.gov/view_text_file.php?"
            f"filename={station}{suffix}{year}.txt.gz"
            f"&dir=data/historical/{kind}/")


def _monthly_archived_url(station: str, year: int, month: int, kind: str) -> str:
    mon = _MONTH_ABBR[month]
    return (f"https://www.ndbc.noaa.gov/view_text_file.php?"
            f"filename={station}{month}{year}.txt.gz"
            f"&dir=data/{kind}/{mon}/")


def _monthly_rolling_url(station: str, month: int, kind: str) -> str:
    mon = _MONTH_ABBR[month]
    return f"https://www.ndbc.noaa.gov/data/{kind}/{mon}/{station}.txt"


# ---------------------------------------------------------------------------
# Header-format parser
# ---------------------------------------------------------------------------

def _parse_archive_text(text: str) -> dict:
    """Parse an NDBC historical/monthly spectral file.

    Layout:
        #YY  MM DD hh mm  .0200  .0325  .0375 ...
        2025 01 01 00 10  0.00   0.00   0.00  ...

    Older yearly files (pre-2007) use a 2-line header with a #units row
    and 2-digit year — handle both. Returns:
        {iso_timestamp_with_offset: [(freq, value), ...], ...}

    The freq parsing is robust against header rows that contain stray
    tokens (e.g. 'YY MM DD hh mm' followed by either decimals like
    '.0200' or two-token formats — all variants seen are handled).
    """
    if not text:
        return {}
    # Reject HTML responses (NDBC sometimes returns an error page with HTTP 200).
    if text.lstrip().startswith("<"):
        return {}

    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return {}

    header_lines = []
    data_start = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#"):
            header_lines.append(ln)
        else:
            data_start = i
            break

    if not header_lines:
        return {}

    # First header line: '#YY MM DD hh mm <freq1> <freq2> ...'
    h0 = header_lines[0].lstrip("#").split()
    # Skip the time columns. The number of leading time columns is
    # variable: usually 5 (YY MM DD hh mm) but some legacy files don't
    # have the minute column. Detect by finding the first token that
    # parses as a frequency (positive float < 1.0).
    time_col_count = 0
    for tok in h0:
        try:
            f = float(tok)
            if 0.0 < f < 1.0:
                break
        except ValueError:
            pass
        time_col_count += 1
    freqs: list[float] = []
    for tok in h0[time_col_count:]:
        try:
            f = float(tok)
            if 0.0 < f < 1.0:
                freqs.append(f)
        except ValueError:
            continue
    if not freqs:
        return {}

    out: dict[str, list[tuple[float, float]]] = {}
    for ln in lines[data_start:]:
        if ln.startswith("#"):
            # A second '#' line in the header (units row in modern files).
            continue
        toks = ln.split()
        if len(toks) < time_col_count + len(freqs):
            continue
        try:
            yr = int(toks[0])
            if yr < 100:  # legacy 2-digit year
                yr += 1900 if yr >= 70 else 2000
            mo = int(toks[1])
            dy = int(toks[2])
            hr = int(toks[3])
            mi = int(toks[4]) if time_col_count >= 5 else 0
            t = datetime(yr, mo, dy, hr, mi, tzinfo=dtz.utc)
        except (ValueError, IndexError):
            continue
        bins: list[tuple[float, float]] = []
        for k, freq in enumerate(freqs):
            try:
                v = float(toks[time_col_count + k])
            except (ValueError, IndexError):
                continue
            if v in _FILL:
                continue
            bins.append((freq, v))
        if not bins:
            continue
        out[t.strftime("%Y-%m-%dT%H:%M:%SZ")] = bins
    return out


# ---------------------------------------------------------------------------
# Decomposition → row records
# ---------------------------------------------------------------------------

def _components_to_rows(buoy_id: str, ts_iso: str, components: list,
                       source_tag: str, ingest_utc: str) -> list[dict]:
    """Convert _spectral_components output (height_ft, period_s, dir, energy)
    into partition=1+ rows in the shared obs schema (hs in METERS).
    Components are already energy-sorted by `_spectral_components`."""
    rows: list[dict] = []
    for i, c in enumerate(components, start=1):
        h_ft = c.get("height_ft")
        rows.append({
            "buoy_id":    buoy_id,
            "valid_utc":  ts_iso,
            "partition":  i,
            "hs_m":       (float(h_ft) / FT_PER_M) if h_ft is not None else None,
            "tp_s":       c.get("period_s"),
            "dp_deg":     c.get("direction_deg"),
            "source":     source_tag,
            "ingest_utc": ingest_utc,
        })
    return rows


def _decompose_pair(swden_text: str, swdir_text: str, buoy_id: str,
                    source_tag: str, ingest_utc: str,
                    *, year_filter: int | None = None,
                    month_filter: int | None = None) -> list[dict]:
    """Parse both files, align by timestamp, run dashboard decomposition,
    return a flat list of partition rows. Optional year/month filters
    drop rows outside the target window (defensive — monthly rolling
    can spill into adjacent months)."""
    spec_all = _parse_archive_text(swden_text)
    swdir_all = _parse_archive_text(swdir_text)
    if not spec_all:
        return []
    rows: list[dict] = []
    common_keys = sorted(set(spec_all) & set(swdir_all))
    for ts_iso in common_keys:
        if year_filter is not None and not ts_iso.startswith(f"{year_filter:04d}"):
            continue
        if month_filter is not None and ts_iso[5:7] != f"{month_filter:02d}":
            continue
        spec_bins = spec_all[ts_iso]
        swdir_bins = swdir_all[ts_iso]
        components = _spectral_components(spec_bins, swdir_bins)
        if not components:
            continue
        rows.extend(_components_to_rows(buoy_id, ts_iso, components,
                                         source_tag, ingest_utc))
    return rows


# ---------------------------------------------------------------------------
# Fetch with retry
# ---------------------------------------------------------------------------

def _fetch(url: str, *, attempts: int = 3, timeout: int = 60) -> str | None:
    """Fetch with bounded retries. Returns text on 200 with non-empty body,
    None otherwise. Treats HTML bodies (NDBC error pages) as failure."""
    last_status = None
    for k in range(attempts):
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
            last_status = r.status_code
            if r.status_code == 200 and r.text and not r.text.lstrip().startswith("<"):
                return r.text
            if r.status_code in (404, 410):
                return None  # no archive, no point retrying
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


# ---------------------------------------------------------------------------
# Per-(buoy, year) and per-(buoy, year, month) workers
# ---------------------------------------------------------------------------

def _realtime_url(station: str, kind: str) -> str:
    """Realtime2 spectral feed: ~45 days of data, freq-in-parens format.
    Used as a coverage backstop for buoys (44091, 44097, 44098) that NDBC
    rebroadcasts in realtime but doesn't host historical archives for —
    they're maintained by USACE / UNH / NERACOOS."""
    return f"https://www.ndbc.noaa.gov/data/realtime2/{station}.{kind}"


def _shard_realtime(buoy_id: str) -> Path:
    """One realtime shard per buoy, partitioned by valid-utc year folder
    (matched to the latest year present in the realtime window)."""
    today = datetime.now(dtz.utc)
    return (HIST_OBS_DIR / f"buoy={buoy_id}" / f"year={today.year}"
            / "spectral-realtime.parquet")


def _shard_yearly(buoy_id: str, year: int) -> Path:
    return HIST_OBS_DIR / f"buoy={buoy_id}" / f"year={year}" / "spectral.parquet"


def _shard_monthly(buoy_id: str, year: int, month: int) -> Path:
    return (HIST_OBS_DIR / f"buoy={buoy_id}" / f"year={year}"
            / f"spectral-{year:04d}-{month:02d}.parquet")


def _write_shard(out_path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "buoy_id", "valid_utc", "partition",
        "hs_m", "tp_s", "dp_deg", "source", "ingest_utc",
    ])
    # Drop any row missing all three swell scalars — useless as training data.
    df = df.dropna(subset=["hs_m", "tp_s", "dp_deg"], how="all")
    if df.empty:
        return 0
    # Defensive: dedupe (valid_utc, partition) — should already be unique.
    df = df.sort_values("ingest_utc").drop_duplicates(
        subset=["valid_utc", "partition"], keep="last"
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(out_path)
    return len(df)


def process_buoy_year(buoy_id: str, year: int, *, force: bool = False) -> dict:
    """Top-level worker: fetch yearly swden+swdir, decompose, write shard.
    Returns a small status dict for the caller's report."""
    out_path = _shard_yearly(buoy_id, year)
    status = {"buoy_id": buoy_id, "year": year, "rows": 0,
              "skipped": False, "missing": False, "error": None,
              "out": str(out_path)}
    if out_path.exists() and not force:
        try:
            status["rows"] = len(pd.read_parquet(out_path, columns=["valid_utc"]))
            status["skipped"] = True
            return status
        except Exception:
            pass

    swden = _fetch(_yearly_url(buoy_id, year, "swden"))
    swdir = _fetch(_yearly_url(buoy_id, year, "swdir"))
    if swden is None or swdir is None:
        status["missing"] = True
        return status

    ingest_utc = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        rows = _decompose_pair(swden, swdir, buoy_id,
                                source_tag="ndbc_spectral_archive",
                                ingest_utc=ingest_utc,
                                year_filter=year)
        status["rows"] = _write_shard(out_path, rows)
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
    return status


def process_buoy_realtime(buoy_id: str, *, force: bool = False) -> dict:
    """Realtime-window worker: fetch realtime2 .data_spec + .swdir (~45
    days, freq-in-parens format), decompose every timestamp, write to
    `.csc_data/observations/buoy=<id>/year=<current>/spectral-realtime.parquet`.

    The realtime feed is the only spectral source for buoys NDBC doesn't
    archive (44091/44097/44098) and is also a useful backstop for the
    archived buoys (it covers the gap between the latest published
    yearly file and today). Decomposition is byte-identical to the
    yearly path because both call `_spectral_components`."""
    out_path = _shard_realtime(buoy_id)
    status = {"buoy_id": buoy_id, "realtime": True, "rows": 0,
              "skipped": False, "missing": False, "error": None,
              "out": str(out_path)}
    if out_path.exists() and not force:
        try:
            status["rows"] = len(pd.read_parquet(out_path, columns=["valid_utc"]))
            status["skipped"] = True
            return status
        except Exception:
            pass

    swden = _fetch(_realtime_url(buoy_id, "data_spec"))
    swdir = _fetch(_realtime_url(buoy_id, "swdir"))
    if swden is None or swdir is None:
        status["missing"] = True
        return status

    try:
        # Realtime files use freq-in-parens; reuse the dashboard's parser.
        # value_offset=1 for .data_spec (sep_freq column), 0 for .swdir.
        spec_all = _parse_spectral_file_all_rows(swden, value_offset=1)
        swdir_all = _parse_spectral_file_all_rows(swdir, value_offset=0)
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        return status
    if not spec_all:
        status["missing"] = True
        return status

    ingest_utc = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # The realtime parser returns ISO timestamps with a +00:00 offset (no
    # `Z` suffix). Normalize to the shared canonical form.
    def _norm_iso(s: str) -> str:
        # Cheap path: replace '+00:00' tail with 'Z'.
        if s.endswith("+00:00"):
            return s[:-6] + "Z"
        return s

    rows: list[dict] = []
    common_keys = sorted(set(spec_all) & set(swdir_all))
    for ts_key in common_keys:
        spec_bins = spec_all[ts_key]
        swdir_bins = swdir_all[ts_key]
        components = _spectral_components(spec_bins, swdir_bins)
        if not components:
            continue
        rows.extend(_components_to_rows(buoy_id, _norm_iso(ts_key),
                                         components,
                                         "ndbc_spectral_realtime",
                                         ingest_utc))

    # Realtime can span Dec→Jan: distribute rows into per-year shards so
    # the obs reader picks them up under the right year=Y folder.
    by_year: dict[int, list[dict]] = {}
    for r in rows:
        try:
            yr = int(r["valid_utc"][:4])
        except Exception:
            continue
        by_year.setdefault(yr, []).append(r)

    total = 0
    for yr, yr_rows in by_year.items():
        path = (HIST_OBS_DIR / f"buoy={buoy_id}" / f"year={yr}"
                / "spectral-realtime.parquet")
        total += _write_shard(path, yr_rows)
    status["rows"] = total
    return status


def process_buoy_month(buoy_id: str, year: int, month: int,
                       *, force: bool = False) -> dict:
    """Monthly worker: tries archived → rolling. Skips current month."""
    out_path = _shard_monthly(buoy_id, year, month)
    status = {"buoy_id": buoy_id, "year": year, "month": month, "rows": 0,
              "skipped": False, "missing": False, "error": None,
              "out": str(out_path)}
    today = datetime.now(dtz.utc)
    if (year, month) >= (today.year, today.month):
        # Current/future month has no archived/rolling source; obs_logger
        # is the canonical writer for the current month.
        status["missing"] = True
        return status

    if out_path.exists() and not force:
        try:
            status["rows"] = len(pd.read_parquet(out_path, columns=["valid_utc"]))
            status["skipped"] = True
            return status
        except Exception:
            pass

    candidates: list[tuple[str, str, str]] = [
        ("ndbc_spectral_monthly_archive",
         _monthly_archived_url(buoy_id, year, month, "swden"),
         _monthly_archived_url(buoy_id, year, month, "swdir")),
        ("ndbc_spectral_monthly_rolling",
         _monthly_rolling_url(buoy_id, month, "swden"),
         _monthly_rolling_url(buoy_id, month, "swdir")),
    ]
    swden = swdir = None
    source_tag = ""
    for tag, u_den, u_dir in candidates:
        d = _fetch(u_den)
        s = _fetch(u_dir)
        if d and s:
            swden, swdir, source_tag = d, s, tag
            break
    if swden is None or swdir is None:
        status["missing"] = True
        return status

    ingest_utc = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        rows = _decompose_pair(swden, swdir, buoy_id,
                                source_tag=source_tag,
                                ingest_utc=ingest_utc,
                                year_filter=year, month_filter=month)
        status["rows"] = _write_shard(out_path, rows)
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
    return status


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _parse_month_range(spec: str) -> list[tuple[int, int]]:
    a, b = spec.split(":")
    ay, am = (int(x) for x in a.split("-"))
    by, bm = (int(x) for x in b.split("-"))
    out: list[tuple[int, int]] = []
    y, m = ay, am
    while (y, m) <= (by, bm):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def run_yearly(start_year: int, end_year: int,
               buoy_ids: list[str], *,
               force: bool = False, parallel: int = 4) -> list[dict]:
    ensure_dirs()
    HIST_OBS_DIR.mkdir(parents=True, exist_ok=True)
    label = {b[0]: b[1] for b in BUOYS}
    jobs = [(b, y) for b in buoy_ids for y in range(start_year, end_year + 1)]
    print(f"[spectral_backfill] yearly  buoys={buoy_ids}  "
          f"years={start_year}..{end_year}  jobs={len(jobs)}  "
          f"parallel={parallel}", flush=True)
    results: list[dict] = []
    if parallel <= 1:
        for b, y in jobs:
            r = process_buoy_year(b, y, force=force)
            _print_status(r, label)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            fut_map = {pool.submit(process_buoy_year, b, y, force=force): (b, y)
                       for b, y in jobs}
            for fut in as_completed(fut_map):
                b, y = fut_map[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"buoy_id": b, "year": y, "rows": 0,
                         "skipped": False, "missing": False,
                         "error": f"{type(e).__name__}: {e}",
                         "out": ""}
                _print_status(r, label)
                results.append(r)
    return results


def run_monthly(months: list[tuple[int, int]],
                buoy_ids: list[str], *,
                force: bool = False, parallel: int = 4) -> list[dict]:
    ensure_dirs()
    HIST_OBS_DIR.mkdir(parents=True, exist_ok=True)
    label = {b[0]: b[1] for b in BUOYS}
    jobs = [(b, y, m) for b in buoy_ids for (y, m) in months]
    print(f"[spectral_backfill] monthly buoys={buoy_ids}  "
          f"months={months[0][0]}-{months[0][1]:02d}.."
          f"{months[-1][0]}-{months[-1][1]:02d}  jobs={len(jobs)}  "
          f"parallel={parallel}", flush=True)
    results: list[dict] = []
    if parallel <= 1:
        for b, y, m in jobs:
            r = process_buoy_month(b, y, m, force=force)
            _print_status(r, label)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            fut_map = {pool.submit(process_buoy_month, b, y, m, force=force): (b, y, m)
                       for b, y, m in jobs}
            for fut in as_completed(fut_map):
                b, y, m = fut_map[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"buoy_id": b, "year": y, "month": m, "rows": 0,
                         "skipped": False, "missing": False,
                         "error": f"{type(e).__name__}: {e}",
                         "out": ""}
                _print_status(r, label)
                results.append(r)
    return results


def run_realtime(buoy_ids: list[str], *,
                 force: bool = False, parallel: int = 4) -> list[dict]:
    """Realtime spectral pull for each buoy. ~45 days/buoy."""
    ensure_dirs()
    HIST_OBS_DIR.mkdir(parents=True, exist_ok=True)
    label = {b[0]: b[1] for b in BUOYS}
    print(f"[spectral_backfill] realtime buoys={buoy_ids}  jobs={len(buoy_ids)}  "
          f"parallel={parallel}", flush=True)
    results: list[dict] = []
    if parallel <= 1:
        for b in buoy_ids:
            r = process_buoy_realtime(b, force=force)
            _print_status(r, label)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            fut_map = {pool.submit(process_buoy_realtime, b, force=force): b
                       for b in buoy_ids}
            for fut in as_completed(fut_map):
                b = fut_map[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"buoy_id": b, "realtime": True, "rows": 0,
                         "skipped": False, "missing": False,
                         "error": f"{type(e).__name__}: {e}",
                         "out": ""}
                _print_status(r, label)
                results.append(r)
    return results


def _print_status(r: dict, label: dict) -> None:
    bid = r["buoy_id"]
    name = label.get(bid, "?")
    if r.get("realtime"):
        tag = "realtime"
    elif "month" in r:
        tag = f"{r['year']}-{r['month']:02d}"
    else:
        tag = f"{r['year']}"
    if r.get("error"):
        print(f"  {bid} ({name}) {tag}: ERROR {r['error']}", flush=True)
    elif r.get("missing"):
        print(f"  {bid} ({name}) {tag}: missing (no NDBC archive)", flush=True)
    elif r.get("skipped"):
        print(f"  {bid} ({name}) {tag}: skipped — {r['rows']:>6} rows already on disk",
              flush=True)
    else:
        print(f"  {bid} ({name}) {tag}: {r['rows']:>6} partition rows", flush=True)


def _log_summary(tag: str, results: list[dict], elapsed: float) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(r["rows"] for r in results if not r.get("error"))
    n_err = sum(1 for r in results if r.get("error"))
    n_miss = sum(1 for r in results if r.get("missing"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    line = (f"[{datetime.now(dtz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
            f"{tag}  jobs={len(results)} written={total} "
            f"missing={n_miss} skipped={n_skip} errors={n_err} "
            f"elapsed={elapsed:.1f}s\n")
    with (LOGS_DIR / "ndbc_spectral_backfill.log").open("a") as f:
        f.write(line)
    print(line.rstrip(), flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(prog="csc2.ndbc_spectral_backfill")
    ap.add_argument("--scope", choices=["east", "west", "all"], default="east",
                    help="Default east (matches the user-facing CSC2 track). "
                         "Use 'all' to backfill all 8 CSC2 buoys.")
    ap.add_argument("--buoy", default=None,
                    help="Single buoy id; overrides --scope.")
    ap.add_argument("--start-year", type=int, default=2021)
    ap.add_argument("--end-year", type=int,
                    default=datetime.now(dtz.utc).year - 1,
                    help="Inclusive end year for yearly archives. "
                         "Defaults to last calendar year (the current year's "
                         "yearly archive isn't published until ~Q1 of the year+1).")
    ap.add_argument("--year", type=int, default=None,
                    help="Convenience: single year (sets start==end).")
    ap.add_argument("--monthly-range", default=None,
                    help="YYYY-MM:YYYY-MM. If set, runs monthly backfill in "
                         "ADDITION to yearly (use to fill the current year's "
                         "months that aren't in the yearly archive yet).")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch and overwrite existing shards.")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--no-yearly", action="store_true",
                    help="Skip yearly backfill (only useful with --monthly-range "
                         "or --realtime).")
    ap.add_argument("--realtime", action="store_true",
                    help="Pull realtime2 .data_spec/.swdir (~45 days) for the "
                         "selected buoys. Required for buoys NDBC doesn't "
                         "archive (44091/44097/44098).")
    args = ap.parse_args()

    if args.buoy:
        buoy_ids = [args.buoy]
    elif args.scope == "all":
        buoy_ids = [b[0] for b in BUOYS]
    else:
        buoy_ids = [b[0] for b in BUOYS if b[4] == args.scope]
    if not buoy_ids:
        print("No buoys selected.", flush=True)
        return 1

    if args.year is not None:
        start_year = end_year = args.year
    else:
        start_year, end_year = args.start_year, args.end_year

    t0 = time.monotonic()
    all_results: list[dict] = []

    if not args.no_yearly:
        yearly = run_yearly(start_year, end_year, buoy_ids,
                            force=args.force, parallel=args.parallel)
        _log_summary(f"yearly {start_year}..{end_year}",
                     yearly, time.monotonic() - t0)
        all_results.extend(yearly)

    if args.monthly_range:
        months = _parse_month_range(args.monthly_range)
        t1 = time.monotonic()
        monthly = run_monthly(months, buoy_ids,
                              force=args.force, parallel=args.parallel)
        _log_summary(f"monthly {args.monthly_range}",
                     monthly, time.monotonic() - t1)
        all_results.extend(monthly)

    if args.realtime:
        t2 = time.monotonic()
        rt = run_realtime(buoy_ids, force=args.force, parallel=args.parallel)
        _log_summary("realtime", rt, time.monotonic() - t2)
        all_results.extend(rt)

    total_rows = sum(r["rows"] for r in all_results if not r.get("error"))
    n_err = sum(1 for r in all_results if r.get("error"))
    print(f"\n[spectral_backfill] DONE  total={total_rows} partition rows  "
          f"errors={n_err}  elapsed={time.monotonic()-t0:.1f}s", flush=True)
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
