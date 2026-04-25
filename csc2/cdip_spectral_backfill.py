"""CSC2 — CDIP THREDDS spectral backfill for the three east-coast buoys
NDBC doesn't archive (44091/97/98 are USACE/UCONN/UNH-owned and only get
realtime relayed by NDBC; CDIP is the canonical archive).

Source: `https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{cdip}p1/
{cdip}p1_historic.nc` — NetCDF / OPeNDAP. Variables:
  waveEnergyDensity(time, freq)  in m²·s   (== m²/Hz; same as NDBC swden)
  waveMeanDirection(time, freq)  in degT   (same as NDBC swdir / alpha1)
  waveFrequency(freq)            in Hz
  waveTime(time)                 datetime64

For each timestamp we run the dashboard's `_spectral_components` (in
`buoy.py`) on the (freq, energy) and (freq, direction) bin pairs to
produce primary/secondary swell partitions, then write rows in the same
schema as the rest of `.csc_data/observations/`. Output:
    .csc_data/observations/buoy=<id>/year=<Y>/spectral-cdip-<Y>.parquet

Usage:
    python -m csc2.cdip_spectral_backfill                # all 3 buoys, all years
    python -m csc2.cdip_spectral_backfill --buoy 44091   # single buoy
    python -m csc2.cdip_spectral_backfill --year 2024
    python -m csc2.cdip_spectral_backfill --force        # re-fetch + overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone as dtz
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csc2.schema import LOGS_DIR, ensure_dirs  # noqa: E402
from buoy import _spectral_components  # noqa: E402

HIST_OBS_DIR = _ROOT / ".csc_data" / "observations"
FT_PER_M = 3.28084

# NDBC ID → CDIP ID. Confirmed by lat/lon matching the metaDeploy* fields.
CDIP_MAP = {
    "44091": "209",  # Barnegat (NJ)        — USACE-owned
    "44097": "154",  # Block Island Sound   — UConn / SeaCAT
    "44098": "160",  # Jeffrey's Ledge (NH) — UNH / NERACOOS
    "46221": "028",  # Santa Monica Bay (CA) — CDIP-operated
    # 46025 (Santa Monica Basin) is NDBC-owned with a swden archive;
    # use csc2.ndbc_spectral_backfill --scope west for it.
}


def _cdip_url(cdip_id: str, mode: str = "historic") -> str:
    """`historic` = QC'd archive (e.g. 2014..2025-06-25 for CDIP 209).
    `realtime` = post-historic rolling window (typically picks up where
    historic ends and runs to ~now). Combining the two = full coverage."""
    if mode == "realtime":
        return (f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/"
                f"{cdip_id}p1_rt.nc")
    return (f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/"
            f"{cdip_id}p1/{cdip_id}p1_historic.nc")


def _shard_path(buoy_id: str, year: int, *, mode: str = "historic") -> Path:
    suffix = "rt" if mode == "realtime" else "hist"
    return (HIST_OBS_DIR / f"buoy={buoy_id}" / f"year={year}"
            / f"spectral-cdip-{suffix}-{year:04d}.parquet")


def _is_finite_pair(f: float, v: float) -> bool:
    return (np.isfinite(f) and np.isfinite(v))


def _decompose_year(buoy_id: str, year: int, freqs: np.ndarray,
                    times: np.ndarray, energy: np.ndarray,
                    direction: np.ndarray, ingest_utc: str) -> list[dict]:
    """Run `_spectral_components` for every timestamp inside `year` and
    flatten the resulting partitions to row dicts."""
    rows: list[dict] = []
    freq_list = [float(f) for f in freqs]
    n_bins = len(freq_list)

    # Mask down to the target year first — much faster than per-row check.
    yrs = pd.DatetimeIndex(times).year.to_numpy()
    in_year = np.where(yrs == year)[0]
    if len(in_year) == 0:
        return rows

    for i in in_year:
        ts = pd.Timestamp(times[i]).strftime("%Y-%m-%dT%H:%M:%SZ")
        e_row = energy[i]; d_row = direction[i]
        spec_bins = []
        dir_bins  = []
        for k in range(n_bins):
            e = e_row[k]; d = d_row[k]
            if _is_finite_pair(freq_list[k], e):
                spec_bins.append((freq_list[k], float(e)))
            if _is_finite_pair(freq_list[k], d):
                dir_bins.append((freq_list[k], float(d)))
        if not spec_bins or not dir_bins:
            continue
        components = _spectral_components(spec_bins, dir_bins)
        for j, c in enumerate(components, start=1):
            h_ft = c.get("height_ft")
            rows.append({
                "buoy_id":    buoy_id,
                "valid_utc":  ts,
                "partition":  j,
                "hs_m":       (float(h_ft) / FT_PER_M) if h_ft is not None else None,
                "tp_s":       c.get("period_s"),
                "dp_deg":     c.get("direction_deg"),
                "source":     "cdip_thredds",
                "ingest_utc": ingest_utc,
            })
    return rows


def process_buoy(buoy_id: str, *, year_filter: int | None = None,
                 force: bool = False, mode: str = "historic") -> dict:
    """Top-level worker: open CDIP {historic,realtime} NetCDF once for this
    buoy, split rows by year, write per-year shards. Returns a status
    summary. `mode='realtime'` covers the gap after `historic.nc` ends
    (typically mid-2025 to today)."""
    if buoy_id not in CDIP_MAP:
        return {"buoy_id": buoy_id, "error": f"no CDIP mapping for {buoy_id}",
                "rows_by_year": {}}

    cdip_id = CDIP_MAP[buoy_id]
    url = _cdip_url(cdip_id, mode=mode)

    # Lazy import so the module is importable on machines without xarray.
    import xarray as xr

    t0 = time.monotonic()
    print(f"[cdip] {buoy_id} (CDIP {cdip_id}) opening {url}", flush=True)
    try:
        ds = xr.open_dataset(url, decode_times=True)
    except Exception as e:
        return {"buoy_id": buoy_id, "error": f"open failed: {type(e).__name__}: {e}",
                "rows_by_year": {}}

    try:
        freqs = ds["waveFrequency"].values
        times = ds["waveTime"].values
        energy = ds["waveEnergyDensity"].values
        direction = ds["waveMeanDirection"].values
    except KeyError as e:
        ds.close()
        return {"buoy_id": buoy_id, "error": f"missing var: {e}",
                "rows_by_year": {}}

    print(f"[cdip] {buoy_id} loaded {len(times):,} timestamps × {len(freqs)} bins "
          f"in {time.monotonic()-t0:.1f}s", flush=True)

    yrs = pd.DatetimeIndex(times).year.to_numpy()
    if year_filter is not None:
        years = [year_filter]
    else:
        years = sorted({int(y) for y in np.unique(yrs)})

    ingest_utc = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows_by_year: dict[int, int] = {}
    for yr in years:
        out_path = _shard_path(buoy_id, yr, mode=mode)
        if out_path.exists() and not force:
            try:
                rows_by_year[yr] = len(pd.read_parquet(out_path, columns=["valid_utc"]))
            except Exception:
                rows_by_year[yr] = 0
            print(f"  {buoy_id} {yr}: skipped — {rows_by_year[yr]:>6} rows on disk",
                  flush=True)
            continue
        t1 = time.monotonic()
        rows = _decompose_year(buoy_id, yr, freqs, times, energy, direction, ingest_utc)
        if not rows:
            rows_by_year[yr] = 0
            print(f"  {buoy_id} {yr}: 0 partitions (no swell-band signal)", flush=True)
            continue
        df = pd.DataFrame(rows, columns=["buoy_id", "valid_utc", "partition",
                                         "hs_m", "tp_s", "dp_deg",
                                         "source", "ingest_utc"])
        df = df.dropna(subset=["hs_m", "tp_s", "dp_deg"], how="all")
        df = df.sort_values("ingest_utc").drop_duplicates(
            subset=["valid_utc", "partition"], keep="last"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        df.to_parquet(tmp, index=False, compression="snappy")
        tmp.replace(out_path)
        rows_by_year[yr] = len(df)
        print(f"  {buoy_id} {yr}: {len(df):>6} partition rows in "
              f"{time.monotonic()-t1:.1f}s", flush=True)

    ds.close()
    elapsed = time.monotonic() - t0
    return {"buoy_id": buoy_id, "cdip_id": cdip_id,
            "rows_by_year": rows_by_year, "elapsed_s": round(elapsed, 1),
            "error": None}


def run(buoy_ids: list[str], *, year_filter: int | None = None,
        force: bool = False, parallel: int = 3,
        mode: str = "historic") -> list[dict]:
    ensure_dirs()
    HIST_OBS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cdip] {mode} backfill buoys={buoy_ids} year={year_filter or 'all'} "
          f"parallel={parallel}", flush=True)
    results: list[dict] = []
    if parallel <= 1:
        for b in buoy_ids:
            results.append(process_buoy(b, year_filter=year_filter, force=force,
                                         mode=mode))
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            fut_map = {pool.submit(process_buoy, b, year_filter=year_filter,
                                    force=force, mode=mode): b for b in buoy_ids}
            for fut in as_completed(fut_map):
                b = fut_map[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"buoy_id": b, "error": f"{type(e).__name__}: {e}",
                         "rows_by_year": {}}
                results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc2.cdip_spectral_backfill")
    ap.add_argument("--buoy", default=None,
                    help="Single buoy id (44091, 44097, or 44098). "
                         "Default: all three.")
    ap.add_argument("--year", type=int, default=None,
                    help="Restrict to one year. Default: every year present.")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch and overwrite existing shards.")
    ap.add_argument("--parallel", type=int, default=3,
                    help="Concurrent buoys (default 3, since CDIP serves "
                         "all three on the same THREDDS host).")
    ap.add_argument("--mode", choices=["historic", "realtime", "both"],
                    default="both",
                    help="`historic` reads `archive/<id>p1/<id>p1_historic.nc` "
                         "(QC'd, ends ~mid-2025). `realtime` reads "
                         "`realtime/<id>p1_rt.nc` (post-historic to ~now). "
                         "`both` runs each in turn so we get full coverage.")
    args = ap.parse_args()

    buoy_ids = [args.buoy] if args.buoy else list(CDIP_MAP.keys())
    t0 = time.monotonic()
    modes = ["historic", "realtime"] if args.mode == "both" else [args.mode]
    results: list[dict] = []
    for m in modes:
        results.extend(run(buoy_ids, year_filter=args.year, force=args.force,
                            parallel=args.parallel, mode=m))

    n_ok = sum(1 for r in results if not r.get("error"))
    total_rows = sum(sum(r.get("rows_by_year", {}).values()) for r in results)
    line = (f"[{datetime.now(dtz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
            f"cdip_backfill  buoys={len(results)} ok={n_ok} "
            f"total_rows={total_rows} elapsed={time.monotonic()-t0:.1f}s\n")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "cdip_spectral_backfill.log").open("a") as f:
        f.write(line)
    print(line.rstrip(), flush=True)
    for r in results:
        if r.get("error"):
            print(f"  {r['buoy_id']}: ERROR {r['error']}", flush=True)
    return 0 if n_ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
