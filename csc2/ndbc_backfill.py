"""CSC2 — NDBC buoy observation historical backfill.

Pulls yearly historical `.txt.gz` files from NDBC's public archive for
each CSC2 buoy, parses them into the same schema the live observation
logger writes, and appends to
    .csc_data/observations/buoy=<id>/year=Y/... (long-format per year)

Only a subset of fields survive this path (buoy-side combined Hs / DPD /
MWD only, no real-time spectral partitions) — NDBC's spectral `.swdirN`
archive lives at NDBC THREDDS and is a separate, heavier pull. This
backfill focuses on the cheap yearly text archive so we have a dense
multi-year record of what the buoys actually observed.

Archive layout:
    https://www.ndbc.noaa.gov/view_text_file.php?filename=<sta>h<year>.txt.gz
        &dir=data/historical/stdmet/

Usage:
    python -m csc2.ndbc_backfill                      # all buoys × 2019..today
    python -m csc2.ndbc_backfill --start-year 2021    # custom start
    python -m csc2.ndbc_backfill --buoy 44065         # one buoy
    python -m csc2.ndbc_backfill --force              # overwrite existing
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone as dtz
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csc2.schema import BUOYS, LOGS_DIR, ensure_dirs  # noqa: E402

HIST_OBS_DIR = _ROOT / ".csc_data" / "observations"


def _yearly_url(station_id: str, year: int) -> str:
    return (f"https://www.ndbc.noaa.gov/view_text_file.php?"
            f"filename={station_id}h{year}.txt.gz"
            f"&dir=data/historical/stdmet/")


def _parse_stdmet(text: str) -> pd.DataFrame:
    """Parse NDBC stdmet yearly archive text into a DataFrame with
    columns: valid_utc (str), hs_m, tp_s, dp_deg.
    Files starting from 2007 use a 2-line header (vars + units) and YY=YYYY."""
    lines = [ln for ln in text.splitlines() if ln]
    if not lines:
        return pd.DataFrame()

    # Header detection: line 0 starts with '#YY' for modern files.
    hdr = lines[0].lstrip("#").split()
    # Units line is line 1 if it starts with '#' in modern files.
    data_start = 1
    if len(lines) > 1 and lines[1].startswith("#"):
        data_start = 2

    def col(name):
        try:
            return hdr.index(name)
        except ValueError:
            return None

    idx_year = col("YYYY") if col("YYYY") is not None else col("YY")
    idx_mm = col("MM"); idx_dd = col("DD"); idx_hh = col("hh")
    idx_mn = col("mm") if "mm" in hdr else None
    idx_wvht = col("WVHT")
    idx_dpd = col("DPD")
    idx_mwd = col("MWD")

    if idx_year is None or idx_wvht is None:
        return pd.DataFrame()

    rows = []
    for ln in lines[data_start:]:
        tok = ln.split()
        if not tok: continue
        try:
            yr = int(tok[idx_year])
            if yr < 100:   # legacy 2-digit
                yr += 1900 if yr >= 70 else 2000
            mo = int(tok[idx_mm]); dy = int(tok[idx_dd]); hr = int(tok[idx_hh])
            mi = int(tok[idx_mn]) if idx_mn is not None else 0
            t = datetime(yr, mo, dy, hr, mi, tzinfo=dtz.utc)
        except (ValueError, IndexError):
            continue
        def _f(i):
            if i is None: return None
            try:
                v = float(tok[i])
                return None if v in (99.0, 99, 999.0, 999, 9999.0, 9999) else v
            except (ValueError, IndexError):
                return None
        rows.append({
            "valid_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hs_m":      _f(idx_wvht),
            "tp_s":      _f(idx_dpd),
            "dp_deg":    _f(idx_mwd),
        })
    return pd.DataFrame(rows)


def _backfill_buoy_year(station_id: str, year: int, *, force: bool) -> int:
    """Fetch one (buoy, year), write to {HIST_OBS_DIR}/buoy=<sta>/year=<yr>.parquet."""
    out_dir = HIST_OBS_DIR / f"buoy={station_id}" / f"year={year}"
    out_path = out_dir / "stdmet.parquet"
    if out_path.exists() and not force:
        try:
            return len(pd.read_parquet(out_path))
        except Exception:
            pass

    url = _yearly_url(station_id, year)
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "ColeSurfs/CSC2"})
        if r.status_code != 200 or not r.text.strip() or r.text.lstrip().startswith("<"):
            return 0
    except Exception:
        return 0
    df = _parse_stdmet(r.text)
    if df.empty:
        return 0

    df.insert(0, "buoy_id", station_id)
    df["partition"] = 0
    df["source"]    = "ndbc_stdmet_archive"
    df["ingest_utc"] = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Column order matches the historical observations schema on disk
    df = df[["buoy_id", "valid_utc", "partition",
              "hs_m", "tp_s", "dp_deg", "source", "ingest_utc"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    return len(df)


def run_backfill(*, start_year: int, end_year: int,
                  buoy_ids: list[str] | None = None,
                  force: bool = False, parallel: int = 8) -> dict:
    ensure_dirs()
    HIST_OBS_DIR.mkdir(parents=True, exist_ok=True)
    targets = buoy_ids or [b[0] for b in BUOYS]
    t0 = time.monotonic()
    report: dict[str, dict[int, int]] = {b: {} for b in targets}
    label_by_id = {b[0]: b[1] for b in BUOYS}

    # NDBC stdmet archives are ~1 MB gzipped, and the server is comfortable
    # with modest parallelism. Fan yearly fetches out across a thread pool
    # so 56 (buoys × years) pulls finish in the time of ~7.
    jobs: list[tuple[str, int]] = [(bid, yr)
                                   for bid in targets
                                   for yr in range(start_year, end_year + 1)]
    if parallel <= 1:
        for bid, yr in jobs:
            n = _backfill_buoy_year(bid, yr, force=force)
            report[bid][yr] = n
            print(f"  {bid} {yr}: {n:>6} rows", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            fut_to_job = {pool.submit(_backfill_buoy_year, bid, yr, force=force):
                          (bid, yr) for bid, yr in jobs}
            for fut in as_completed(fut_to_job):
                bid, yr = fut_to_job[fut]
                try:
                    n = fut.result()
                except Exception as e:
                    n = 0
                    print(f"  {bid} {yr}: FAILED ({type(e).__name__}: {e})",
                          flush=True)
                report[bid][yr] = n
                print(f"  {bid} ({label_by_id.get(bid,'?')}) {yr}: "
                      f"{n:>6} rows", flush=True)

    elapsed = time.monotonic() - t0
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "ndbc_backfill.log").open("a") as f:
        f.write(f"[{datetime.now(dtz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
                f"range {start_year}..{end_year}  "
                f"total_rows={sum(sum(v.values()) for v in report.values())}  "
                f"elapsed={elapsed:.1f}s\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc2.ndbc_backfill")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int,
                    default=datetime.now(dtz.utc).year)
    ap.add_argument("--buoy", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--parallel", type=int, default=8,
                    help="Concurrent yearly fetches (default 8).")
    args = ap.parse_args()
    ids = [args.buoy] if args.buoy else None
    report = run_backfill(start_year=args.start_year, end_year=args.end_year,
                           buoy_ids=ids, force=args.force,
                           parallel=args.parallel)
    total = sum(sum(v.values()) for v in report.values())
    print(f"\n[ndbc_backfill] done: {total} rows across {len(report)} buoys, "
          f"years {args.start_year}..{args.end_year}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
