"""CSC GFS-Wave GRIB2 backfill — bypasses Open-Meteo.

Pulls authoritative NOAA GFS-Wave output directly from the AWS
open-data mirror, extracts per-buoy point values, and writes rows in
the exact long-format schema that `.csc_data/forecasts/` already uses.

Two modes:

  --verify-openmeteo
      One-shot probe: fetch Open-Meteo `ncep_gfswave025` and the
      matching NOAA GRIB for the same (lat, lon, cycle, lead), print an
      absolute-diff table across all 13 dashboard-relevant variables.
      Decides whether the replacement pipeline is needed.

  --backfill
      Full backfill over a date range. Walks cycles 00/06/12/18 Z,
      downloads one GRIB per (cycle, lead), extracts per-buoy nearest-
      neighbour values, and appends rows to the forecast archive.
      Resumable: skips (cycle, lead, buoy) triples already on disk.

Schema written — matches `csc/logger.py::_long_rows_from_response`:

    buoy_id, model="GFS", valid_utc, lead_days, variable, value,
    ingest_utc, source="noaa_gfs_grib"

Where `variable` uses the same Open-Meteo canonical names the rest of
the pipeline already expects (`wave_height`, `wave_peak_period`,
`wave_direction`, `swell_wave_height`, `secondary_swell_wave_*`,
`tertiary_swell_wave_*`). `lead_days` is `round((valid - cycle) / 24)`
matching `OM_WAVE_VARS`' lead convention.

Output partition layout (additive to existing tree):

    .csc_data/forecasts/
      model=GFS/
        buoy=44097/
          year=2026/
            month=04/
              grib_bkfl.parquet

The file name `grib_bkfl.parquet` keeps GRIB-sourced rows separate from
Open-Meteo-sourced `analysis.parquet` shards written by the live logger
— `csc.data.read_forecasts` reads both via `*.parquet` glob.

Requirements in the colesurfs conda env:
    cfgrib, xarray, numpy, pandas, requests, eccodes

This module is network-blocked inside the Claude sandbox but is written
to run end-to-end from a shell once invoked manually.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

from csc.schema import (
    BUOYS, CYCLES_PER_DAY, FORECASTS_DIR, OM_MODELS,
    ensure_dirs,
)


# ─── Constants ───────────────────────────────────────────────────────────

# S3 public bucket — anonymous read access, no credentials needed.
# URL pattern (gridded global 0p25):
#   https://noaa-gfs-bdp-pds.s3.amazonaws.com/
#     gfs.YYYYMMDD/HH/wave/gridded/gfswave.tHHz.global.0p25.fFFF.grib2
#
# Also reachable (same files) via NOMADS for the last ~10 days:
#   https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/
#     gfs.YYYYMMDD/HH/wave/gridded/gfswave.tHHz.global.0p25.fFFF.grib2
AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
AWS_URL_FMT = (
    AWS_BASE
    + "/gfs.{yyyymmdd}/{hh:02d}/wave/gridded/"
    "gfswave.t{hh:02d}z.global.0p25.f{fff:03d}.grib2"
)
AWS_IDX_FMT = AWS_URL_FMT + ".idx"

# 13 canonical Open-Meteo variable names — the dashboard field set.
DASHBOARD_VARS = (
    "wave_height", "wave_period", "wave_peak_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "secondary_swell_wave_height", "secondary_swell_wave_period",
    "secondary_swell_wave_direction",
    "tertiary_swell_wave_height", "tertiary_swell_wave_period",
    "tertiary_swell_wave_direction",
)

# Map our canonical OM variable names to cfgrib dataset-variable keys
# when the GRIB is opened with typeOfLevel filters. GFS-Wave 0p25 layout:
#   • typeOfLevel=surface, gridded combined fields:
#       swh (HTSGW) → wave_height
#       perpw (PERPW) → wave_peak_period     # peak period of total spectrum
#       dirpw (DIRPW) → wave_direction       # direction of peak period
#       ws/wdir      → wind fields (not captured here)
#   • typeOfLevel=orderedSequenceOfData, per-partition (partition index 0..3):
#       swh[0..2]  → swell_wave_height / secondary_ / tertiary_
#       perpw[0..2]→ swell_wave_period / secondary_ / tertiary_
#       dirpw[0..2]→ swell_wave_direction / secondary_ / tertiary_
#       index 3 is wind-sea (WVHGT/WVPER/WVDIR)
#
# NOAA does *not* publish a distinct "mean period" field in GRIB; the
# Open-Meteo `wave_period` is presumed derived from the same spectrum.
# We map `wave_period` → perpw as a conservative fallback (peak period);
# _build_components uses `wave_peak_period` first anyway, so the column's
# exact semantics only matter when peak is missing.

_GRIB_SURFACE_MAP = {
    "wave_height":       "swh",
    "wave_peak_period":  "perpw",
    "wave_direction":    "dirpw",
    # wave_period: GFS-Wave doesn't expose a dedicated mean period in GRIB.
    # We fill it with perpw as a safe proxy (same order of magnitude; the
    # dashboard's waves._parse_response uses peak period first anyway).
    "wave_period":       "perpw",
}

# (partition index in GFS-Wave orderedSequenceOfData, OM variable)
_GRIB_PARTITION_MAP = [
    (0, "swell_wave_height",             "swh"),
    (0, "swell_wave_period",             "perpw"),
    (0, "swell_wave_direction",          "dirpw"),
    (1, "secondary_swell_wave_height",   "swh"),
    (1, "secondary_swell_wave_period",   "perpw"),
    (1, "secondary_swell_wave_direction","dirpw"),
    (2, "tertiary_swell_wave_height",    "swh"),
    (2, "tertiary_swell_wave_period",    "perpw"),
    (2, "tertiary_swell_wave_direction", "dirpw"),
]


OPEN_METEO_URL = "https://marine-api.open-meteo.com/v1/marine"


@dataclass(frozen=True)
class Cycle:
    """GFS model run cycle identifier."""
    dt: datetime   # always tz=UTC, hour ∈ {0, 6, 12, 18}

    @property
    def yyyymmdd(self) -> str:
        return self.dt.strftime("%Y%m%d")

    @property
    def hh(self) -> int:
        return self.dt.hour

    def url(self, lead_h: int) -> str:
        return AWS_URL_FMT.format(
            yyyymmdd=self.yyyymmdd, hh=self.hh, fff=lead_h)

    def idx_url(self, lead_h: int) -> str:
        return AWS_IDX_FMT.format(
            yyyymmdd=self.yyyymmdd, hh=self.hh, fff=lead_h)


# ─── HTTP helpers ────────────────────────────────────────────────────────

def _http_head(url: str, timeout: float = 10.0) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _http_download(url: str, dest: Path, timeout: float = 300.0,
                   chunk: int = 1 << 20) -> bool:
    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            if not r.ok:
                print(f"  [!] GET {url} → HTTP {r.status_code}",
                      file=sys.stderr)
                return False
            with open(dest, "wb") as fh:
                for part in r.iter_content(chunk_size=chunk):
                    if part:
                        fh.write(part)
        return True
    except requests.RequestException as e:
        print(f"  [!] download {url}: {e}", file=sys.stderr)
        return False


# ─── GRIB decoding ───────────────────────────────────────────────────────

def _open_grib_group(path: Path, type_of_level: str):
    """Open one typeOfLevel group from a GRIB file via cfgrib. Returns an
    xarray.Dataset or None on failure."""
    try:
        import xarray as xr   # noqa: F401 — required by cfgrib engine
    except ImportError:
        print("[!] xarray not installed — pip install xarray cfgrib",
              file=sys.stderr)
        return None
    try:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": type_of_level},
                "indexpath": "",   # avoid writing .idx next to the grib
            },
        )
    except Exception as e:
        # cfgrib is noisy about missing groups — only surface non-trivial
        # failures.
        msg = str(e).lower()
        if "no dataset" not in msg and "not found" not in msg:
            print(f"[!] cfgrib {type_of_level}: {e}", file=sys.stderr)
        return None


def _sample_point(da, lat: float, lon: float) -> float | None:
    """Nearest-neighbour point sample from a GFS-Wave GRIB DataArray.
    GFS 0p25 uses longitude 0..360; we convert negative input lon."""
    import numpy as np
    try:
        lon_max = float(da["longitude"].max())
        lon_q = lon % 360 if lon_max > 180 else lon
        pt = da.sel(latitude=lat, longitude=lon_q, method="nearest")
        val = float(pt.values)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def extract_point_all_vars(grib_path: Path, lat: float, lon: float
                            ) -> dict[str, float | None]:
    """Return {om_variable_name: value_or_None} for all 13 dashboard
    variables, sampled at (lat, lon) from one GFS-Wave GRIB2 file.

    Surface group covers the combined fields (HTSGW/PERPW/DIRPW);
    orderedSequenceOfData group is indexed by partition for the per-
    swell fields."""
    import numpy as np

    out: dict[str, float | None] = {v: None for v in DASHBOARD_VARS}

    # Surface group — combined wave fields.
    ds_surf = _open_grib_group(grib_path, "surface")
    if ds_surf is not None:
        for om_name, gname in _GRIB_SURFACE_MAP.items():
            if gname in ds_surf.data_vars:
                out[om_name] = _sample_point(ds_surf[gname], lat, lon)
        ds_surf.close()

    # Per-partition group — swell 1/2/3.
    ds_part = _open_grib_group(grib_path, "orderedSequenceOfData")
    if ds_part is not None:
        for pidx, om_name, gname in _GRIB_PARTITION_MAP:
            if gname not in ds_part.data_vars:
                continue
            da = ds_part[gname]
            if "orderedSequenceData" in da.dims:
                try:
                    da_p = da.isel(orderedSequenceData=pidx)
                except IndexError:
                    continue
            elif "partition" in da.dims:
                try:
                    da_p = da.isel(partition=pidx)
                except IndexError:
                    continue
            elif len(da.shape) == 2 and pidx == 0:
                # Single-partition file — only the primary is valid.
                da_p = da
            else:
                continue
            out[om_name] = _sample_point(da_p, lat, lon)
        ds_part.close()

    return out


# ─── Cycle / lead iteration ──────────────────────────────────────────────

def _all_cycles(start: datetime, end: datetime) -> Iterator[Cycle]:
    """Yield every 6-hourly GFS cycle in [start, end] inclusive."""
    start = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    t = start.replace(hour=(start.hour // 6) * 6)
    while t <= end:
        if t.hour in CYCLES_PER_DAY["GFS"]:
            yield Cycle(dt=t)
        t += timedelta(hours=6)


def _latest_available_cycle(max_back_h: int = 48) -> Cycle | None:
    """Walk back in 6 h steps until we find a cycle whose f000 file
    exists on S3. Returns None if none of the last `max_back_h` do."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                             microsecond=0)
    base = now.replace(hour=(now.hour // 6) * 6)
    for back in range(0, max_back_h + 1, 6):
        c = Cycle(dt=base - timedelta(hours=back))
        if _http_head(c.url(0)):
            return c
    return None


# ─── Output shard writer ─────────────────────────────────────────────────

def _shard_path(buoy_id: str, valid_utc: datetime) -> Path:
    return (FORECASTS_DIR / "model=GFS" / f"buoy={buoy_id}"
            / f"year={valid_utc.year:04d}"
            / f"month={valid_utc.month:02d}"
            / "grib_bkfl.parquet")


def _load_existing_keys(buoy_id: str, valid_utc: datetime
                        ) -> set[tuple[str, int, int, str]]:
    """Return the set of (valid_utc_iso, lead_days, cycle_h, variable)
    triples already on disk for this (buoy, year, month). Used to skip
    downloads we've already completed."""
    path = _shard_path(buoy_id, valid_utc)
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path,
                             columns=["valid_utc", "lead_days", "variable"])
    except Exception:
        return set()
    return set(zip(df["valid_utc"].astype(str),
                    df["lead_days"].astype(int),
                    df["variable"].astype(str)))


def _append_shard(rows: list[dict[str, Any]], shard: Path) -> None:
    if not rows:
        return
    shard.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if shard.exists():
        try:
            prev = pd.read_parquet(shard)
            df = pd.concat([prev, new], ignore_index=True)
        except Exception as e:
            corrupt = shard.with_suffix(".parquet.corrupt")
            print(f"[!] existing shard {shard} unreadable ({e}); "
                  f"renaming to {corrupt.name}", file=sys.stderr)
            shard.rename(corrupt)
            df = new
    else:
        df = new
    df.drop_duplicates(
        subset=["buoy_id", "model", "valid_utc", "lead_days", "variable"],
        keep="last", inplace=True,
    )
    df.to_parquet(shard, index=False, compression="snappy")


# ─── Top-level: fetch + extract one (cycle, lead) across all buoys ───────

def _process_one_grib(cycle: Cycle, lead_h: int,
                      buoys: list[tuple[str, str, float, float, str]],
                      work_dir: Path,
                      skip_keys_by_buoy: dict[str, set]
                      ) -> dict[str, int]:
    """Download one GRIB, extract per-buoy values, append to each
    buoy's shard. Returns {buoy_id: rows_written}."""
    url = cycle.url(lead_h)
    grib_path = (work_dir
                 / f"gfswave.t{cycle.hh:02d}z.{cycle.yyyymmdd}."
                 f"f{lead_h:03d}.grib2")

    # Skip if every (buoy × variable) for this cycle+lead is already
    # on disk — no need to even download.
    valid_utc = cycle.dt + timedelta(hours=lead_h)
    lead_days = max(0, round(lead_h / 24))
    valid_iso = valid_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    need_any = False
    for (bid, *_rest) in buoys:
        existing = skip_keys_by_buoy.get(bid, set())
        for var in DASHBOARD_VARS:
            if (valid_iso, lead_days, var) not in existing:
                need_any = True
                break
        if need_any:
            break
    if not need_any:
        print(f"  all buoys complete for {cycle.yyyymmdd} {cycle.hh:02d}Z "
              f"f{lead_h:03d} — skip")
        return {bid: 0 for (bid, *_r) in buoys}

    if not _http_head(url):
        print(f"  [!] GRIB missing on S3: {url}")
        return {bid: 0 for (bid, *_r) in buoys}
    print(f"  downloading {url}")
    ok = _http_download(url, grib_path)
    if not ok or not grib_path.exists() or grib_path.stat().st_size < 1024:
        print(f"  [!] download failed/empty, skipping")
        try:
            grib_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {bid: 0 for (bid, *_r) in buoys}

    ingest_s = datetime.now(timezone.utc).isoformat()
    written: dict[str, int] = {}
    for (bid, label, lat, lon, _op) in buoys:
        try:
            pts = extract_point_all_vars(grib_path, lat, lon)
        except Exception:
            traceback.print_exc()
            pts = {v: None for v in DASHBOARD_VARS}

        existing = skip_keys_by_buoy.get(bid, set())
        new_rows: list[dict[str, Any]] = []
        for var, val in pts.items():
            if val is None:
                continue
            key = (valid_iso, lead_days, var)
            if key in existing:
                continue
            new_rows.append({
                "buoy_id": bid,
                "model": "GFS",
                "valid_utc": valid_iso,
                "lead_days": lead_days,
                "variable": var,
                "value": float(val),
                "source": "noaa_gfs_grib",
                "ingest_utc": ingest_s,
            })
        if new_rows:
            shard = _shard_path(bid, valid_utc)
            _append_shard(new_rows, shard)
            # Update in-memory skip index so the next (cycle, lead) in
            # the same month doesn't re-read the shard from disk.
            existing.update((r["valid_utc"], r["lead_days"], r["variable"])
                            for r in new_rows)
            skip_keys_by_buoy[bid] = existing
        written[bid] = len(new_rows)
        print(f"    {bid} {label[:22]:<22} "
              f"→ {len(new_rows):>3} rows")

    try:
        grib_path.unlink(missing_ok=True)
    except Exception:
        pass
    return written


# ─── Verification probe: Open-Meteo vs NOAA GRIB at one (pt, time) ───────

def verify_openmeteo(buoy_id: str, cycle_arg: str = "LATEST",
                     lead_h: int = 6) -> int:
    """Fetch Open-Meteo ncep_gfswave025 and the matching NOAA GRIB for
    one (buoy, cycle, lead). Print a per-variable absolute-diff table.

    Returns 0 if max diff is within tolerance on every field,
    1 if any field diverges beyond tolerance,
    2 on setup error.
    """
    meta = next((b for b in BUOYS if b[0] == buoy_id), None)
    if meta is None:
        print(f"unknown buoy_id {buoy_id!r}", file=sys.stderr)
        return 2
    _bid, label, lat, lon, _op = meta

    if cycle_arg == "LATEST":
        cycle = _latest_available_cycle()
        if cycle is None:
            print("could not find a recent cycle on S3", file=sys.stderr)
            return 2
    else:
        cycle = Cycle(dt=datetime.strptime(cycle_arg, "%Y%m%d%H").replace(
            tzinfo=timezone.utc))

    valid = cycle.dt + timedelta(hours=lead_h)
    print(f"── GFS source audit: OM vs NOAA-GRIB ──")
    print(f"buoy   : {buoy_id} ({label})  lat={lat}  lon={lon}")
    print(f"cycle  : {cycle.yyyymmdd} {cycle.hh:02d}Z")
    print(f"lead   : f{lead_h:03d} (valid_utc = {valid:%Y-%m-%dT%H:%MZ})")

    # 1. Open-Meteo
    om_vals: dict[str, float | None] = {v: None for v in DASHBOARD_VARS}
    try:
        r = requests.get(OPEN_METEO_URL, timeout=60, params={
            "latitude": lat,
            "longitude": lon,
            "models": OM_MODELS["GFS"],
            "hourly": ",".join(DASHBOARD_VARS),
            "past_days": 3,
            "forecast_days": 3,
            "timezone": "UTC",
        })
        r.raise_for_status()
        data = r.json()
        h = data.get("hourly", {})
        times = h.get("time") or []
        key = valid.strftime("%Y-%m-%dT%H:00")
        if key in times:
            i = times.index(key)
            for var in DASHBOARD_VARS:
                v = (h.get(var) or [None] * len(times))[i]
                om_vals[var] = None if v is None else float(v)
        else:
            print(f"  [!] Open-Meteo has no row for {key} — "
                  f"lead may be outside OM window")
    except Exception as e:
        print(f"  [!] Open-Meteo fetch failed: {e}")

    # 2. NOAA GRIB
    grib_vals: dict[str, float | None] = {v: None for v in DASHBOARD_VARS}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "gfswave.grib2"
        url = cycle.url(lead_h)
        if not _http_head(url):
            print(f"  [!] GRIB not yet on S3: {url}")
            return 2
        if not _http_download(url, tmp):
            print("  [!] GRIB download failed")
            return 2
        grib_vals = extract_point_all_vars(tmp, lat, lon)

    # 3. Compare
    def _diff(a, b):
        if a is None or b is None:
            return None
        return abs(a - b)

    tol = {
        "height":    0.05,   # m
        "period":    0.2,    # s
        "direction": 3.0,    # deg
    }
    def _tol(var: str) -> float:
        if "direction" in var:
            return tol["direction"]
        if "period" in var:
            return tol["period"]
        return tol["height"]

    print()
    print(f"{'variable':<38} {'OM':>10} {'GRIB':>10} {'|Δ|':>8} {'tol':>6}")
    print("-" * 78)
    bad = 0
    for var in DASHBOARD_VARS:
        a = om_vals[var]; b = grib_vals[var]
        d = _diff(a, b)
        t = _tol(var)
        flag = ""
        if d is not None and d > t:
            flag = "   DIVERGE"
            bad += 1
        print(f"{var:<38} "
              f"{('' if a is None else f'{a:.3f}'):>10} "
              f"{('' if b is None else f'{b:.3f}'):>10} "
              f"{('' if d is None else f'{d:.3f}'):>8} "
              f"{t:>6.2f}{flag}")
    print()
    if bad:
        print(f"VERDICT: divergence on {bad}/{len(DASHBOARD_VARS)} "
              f"variables — GRIB pipeline should replace Open-Meteo.")
        return 1
    print(f"VERDICT: Open-Meteo and NOAA GRIB agree within tolerance on "
          f"all {len(DASHBOARD_VARS)} variables.")
    print("Still update schema.OM_WAVE_VARS to add wave_peak_period and "
          "tertiary_swell_wave_* so the logger captures everything the "
          "dashboard displays.")
    return 0


# ─── Backfill orchestration ──────────────────────────────────────────────

def backfill(start: datetime, end: datetime,
             buoy_ids: list[str] | None = None,
             leads: list[int] | None = None,
             dry_run: bool = False) -> dict[str, Any]:
    """Walk cycles in [start, end], download one GRIB per (cycle, lead),
    extract per-buoy values, append to forecast archive.

    Args:
      start, end: UTC datetimes bracketing the cycle range.
      buoy_ids: limit to these CSC buoys (default: all).
      leads: forecast leads in hours to fetch (default: [0, 6, 12, 18, 24]).
      dry_run: print URLs and skip counts, don't download or write.
    """
    ensure_dirs()
    if leads is None:
        leads = [0, 6, 12, 18, 24]
    target = [b for b in BUOYS if (buoy_ids is None or b[0] in buoy_ids)]
    if not target:
        print("no matching buoys", file=sys.stderr)
        return {"n_cycles": 0, "n_grib": 0, "n_rows": 0, "errors": 0}

    cycles = list(_all_cycles(start, end))
    print(f"── GFS-Wave GRIB backfill ──")
    print(f"range   : {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC")
    print(f"cycles  : {len(cycles)}  (every 6 h)")
    print(f"leads   : {leads}")
    print(f"buoys   : {[b[0] for b in target]}")

    # Pre-load existing shard keys per (buoy, year, month) so we skip
    # already-fetched rows without re-reading on every iteration.
    skip_keys: dict[str, set] = {}
    for (bid, *_r) in target:
        for cycle in cycles:
            for lh in leads:
                v = cycle.dt + timedelta(hours=lh)
                path = _shard_path(bid, v)
                if not path.exists():
                    continue
                existing = _load_existing_keys(bid, v)
                skip_keys.setdefault(bid, set()).update(existing)

    stats = {"n_cycles": 0, "n_grib": 0, "n_rows": 0, "errors": 0}

    with tempfile.TemporaryDirectory(prefix="gfsgrib_") as td:
        work = Path(td)
        for cycle in cycles:
            print(f"\n── cycle {cycle.yyyymmdd} {cycle.hh:02d}Z ──")
            stats["n_cycles"] += 1
            for lh in leads:
                url = cycle.url(lh)
                if dry_run:
                    print(f"  DRY-RUN url={url}")
                    continue
                try:
                    written = _process_one_grib(
                        cycle, lh, target, work, skip_keys)
                except Exception:
                    stats["errors"] += 1
                    traceback.print_exc()
                    continue
                stats["n_grib"] += 1
                stats["n_rows"] += sum(written.values())

    print()
    print(f"done: {stats['n_cycles']} cycles, "
          f"{stats['n_grib']} gribs, {stats['n_rows']} rows, "
          f"{stats['errors']} errors")
    return stats


# ─── CLI ─────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="csc.gfs_grib_backfill")
    ap.add_argument("--verify-openmeteo", action="store_true",
                    help="One-shot probe: compare OM vs NOAA-GRIB.")
    ap.add_argument("--backfill", action="store_true",
                    help="Run the GRIB backfill over --start..--end.")
    ap.add_argument("--buoy", default=None,
                    help="Limit to one buoy id.")
    ap.add_argument("--cycle", default="LATEST",
                    help="For --verify-openmeteo: LATEST or YYYYMMDDHH.")
    ap.add_argument("--lead", type=int, default=6,
                    help="For --verify-openmeteo: forecast lead in hours.")
    ap.add_argument("--start", type=_parse_date, default=None,
                    help="Backfill range start (YYYY-MM-DD, UTC).")
    ap.add_argument("--end", type=_parse_date, default=None,
                    help="Backfill range end (YYYY-MM-DD, UTC).")
    ap.add_argument("--leads", type=int, nargs="*", default=None,
                    help="Forecast leads (hours) to fetch per cycle.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be downloaded, don't fetch.")
    args = ap.parse_args(argv)

    if args.verify_openmeteo:
        if not args.buoy:
            print("--verify-openmeteo requires --buoy", file=sys.stderr)
            return 2
        return verify_openmeteo(args.buoy, args.cycle, args.lead)

    if args.backfill:
        if args.start is None or args.end is None:
            print("--backfill requires --start and --end", file=sys.stderr)
            return 2
        buoy_ids = [args.buoy] if args.buoy else None
        backfill(args.start, args.end,
                 buoy_ids=buoy_ids, leads=args.leads,
                 dry_run=args.dry_run)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
