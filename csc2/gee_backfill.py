"""CSC2 — CMEMS ANFC forecast backfill via Google Earth Engine.

Sources historical CMEMS forecast cycles from the GEE ImageCollection
    COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H
which is the only known public archive that preserves lead-time-resolved
ECMWF-WAM-equivalent swell-partition forecasts (via images carrying
`run_time`, `observation_time`, `forecast_hours`, `observation_type`).

For each CSC2 buoy we server-side-sample every `observation_type=forecast`
image at the buoy point, paginate the resulting FeatureCollection, reshape
into the same per-cycle parquet layout the live logger writes to, and feed
each cycle's raw rows through waves_cmems.raw_rows_to_hourly_records — so
the backfill output is byte-identical in form to what the live logger and
the main dashboard produce.

Auth: run `earthengine authenticate` once; credentials persist under
~/.config/earthengine/. Pass the Google Cloud project via CSC2_GEE_PROJECT
(default: "colesurfs-project").

Usage:
    python -m csc2.gee_backfill                 # all 8 buoys
    python -m csc2.gee_backfill --buoy 44065    # one buoy
    python -m csc2.gee_backfill --force         # overwrite existing shards
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone as dtz
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ee  # noqa: E402
from csc2.schema import (  # noqa: E402
    BUOYS, FORECAST_COLUMNS, FORECASTS_DIR, LOGS_DIR, ensure_dirs,
)
from csc2.logger import shard_path, records_to_rows, write_rows  # noqa: E402
from waves_cmems import CMEMS_VARS, raw_rows_to_hourly_records  # noqa: E402


GEE_COLLECTION_ID = "COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H"
GEE_PROJECT = os.environ.get("CSC2_GEE_PROJECT", "colesurfs-project")

# getInfo() on a FeatureCollection is capped around 5000 elements per call;
# we paginate manually with offset/limit via toList.
PAGE_SIZE = 5000


def _init_ee() -> None:
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=GEE_PROJECT)


def _cycle_id_from_ms(run_time_ms: int) -> str:
    t = datetime.fromtimestamp(run_time_ms / 1000, tz=dtz.utc)
    return t.strftime("%Y%m%dT%HZ")


def _list_forecast_cycles() -> list[int]:
    """Return sorted run_time (UTC ms) values for every forecast cycle in
    the collection. Cached across buoys since the same cycle list applies
    globally — sampled once, reused by every _sample_cycle call."""
    coll = (ee.ImageCollection(GEE_COLLECTION_ID)
            .filter(ee.Filter.eq("observation_type", "forecast"))
            .filter(ee.Filter.eq("forecast_hours", 3)))   # one image per cycle
    return sorted(set(coll.aggregate_array("run_time").getInfo()))


def _sample_cycle(run_time_ms: int, lat: float, lon: float) -> list[dict]:
    """Pull every image in one forecast cycle sampled at (lat, lon).
    Returns a list of dicts (one per lead step, ~80) with CMEMS band values
    + run_time + observation_time + forecast_hours props. Unsorted."""
    point = ee.Geometry.Point([lon, lat])
    cycle_coll = (ee.ImageCollection(GEE_COLLECTION_ID)
                  .filter(ee.Filter.eq("observation_type", "forecast"))
                  .filter(ee.Filter.eq("run_time", run_time_ms))
                  .select(CMEMS_VARS))

    def sample(img):
        vals = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=10000,
            bestEffort=True,
        )
        props = vals.combine({
            "run_time":         img.get("run_time"),
            "observation_time": img.get("observation_time"),
            "forecast_hours":   img.get("forecast_hours"),
        })
        return ee.Feature(None, props)

    fc = cycle_coll.map(sample).filter(ee.Filter.notNull(["VHM0"]))
    data = fc.getInfo()
    return [f.get("properties", {}) for f in data.get("features", [])]


def _props_to_raw_row(p: dict) -> dict:
    """Convert one GEE-sampled feature to a raw row in the shape
    waves_cmems.raw_rows_to_hourly_records expects."""
    obs_ms = p.get("observation_time")
    if obs_ms is None:
        return {}
    utc = datetime.fromtimestamp(obs_ms / 1000, tz=dtz.utc)
    row = {"utc": utc}
    for v in CMEMS_VARS:
        row[v] = p.get(v)
    return row


def _backfill_buoy(buoy_id: str, lat: float, lon: float,
                     cycle_times_ms: list[int], *,
                     force: bool = False) -> dict:
    t0 = time.monotonic()
    print(f"\n── {buoy_id} ({lat:.3f}, {lon:.3f}) — {len(cycle_times_ms)} cycles ──",
          flush=True)

    ingest_utc = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows_total = 0
    cycles_written = 0
    cycles_skipped = 0
    cycles_failed = 0

    for i, rt_ms in enumerate(cycle_times_ms, 1):
        cycle = _cycle_id_from_ms(int(rt_ms))
        shard = shard_path(buoy_id, "EURO", cycle)
        if shard.exists() and not force:
            cycles_skipped += 1
            continue

        try:
            feats = _sample_cycle(int(rt_ms), lat, lon)
        except Exception as e:
            print(f"  {buoy_id} {cycle}: sample failed — {type(e).__name__}: {e}")
            cycles_failed += 1
            continue

        raw_rows = [_props_to_raw_row(p) for p in feats]
        raw_rows = [r for r in raw_rows if "utc" in r]
        raw_rows.sort(key=lambda r: r["utc"])
        if not raw_rows:
            cycles_failed += 1
            continue

        records = raw_rows_to_hourly_records(raw_rows)
        rows = records_to_rows(records, buoy_id=buoy_id, model="EURO",
                                 cycle_utc=cycle, ingest_utc=ingest_utc)
        n = write_rows(buoy_id, "EURO", cycle, rows)
        rows_total += n
        cycles_written += 1

        # Progress line every 20 cycles so long runs don't feel silent
        if i % 20 == 0 or i == len(cycle_times_ms):
            dt = time.monotonic() - t0
            rate = i / dt if dt > 0 else 0
            eta = (len(cycle_times_ms) - i) / rate if rate > 0 else 0
            print(f"  {buoy_id}: {i}/{len(cycle_times_ms)} cycles "
                  f"({cycles_written} new, {cycles_skipped} skip, {cycles_failed} fail), "
                  f"{rate:.1f} cyc/s, eta {eta/60:.1f}m", flush=True)

    dt_total = time.monotonic() - t0
    print(f"  {buoy_id}: DONE — wrote {cycles_written} cycles "
          f"({cycles_skipped} skipped, {cycles_failed} failed, {rows_total} rows) "
          f"in {dt_total:.1f}s", flush=True)
    return {
        "buoy_id": buoy_id,
        "cycles": cycles_written,
        "cycles_skipped": cycles_skipped,
        "cycles_failed": cycles_failed,
        "rows": rows_total,
        "elapsed_s": round(dt_total, 1),
    }


def run_backfill(buoy_ids: list[str] | None = None, *, force: bool = False):
    ensure_dirs()
    _init_ee()

    print("[gee_backfill] enumerating cycles via GEE…", flush=True)
    t0 = time.monotonic()
    cycles = _list_forecast_cycles()
    print(f"[gee_backfill] {len(cycles)} cycles discovered "
          f"({datetime.fromtimestamp(cycles[0]/1000, tz=dtz.utc)}"
          f" → {datetime.fromtimestamp(cycles[-1]/1000, tz=dtz.utc)}) "
          f"in {time.monotonic()-t0:.1f}s", flush=True)

    targets = buoy_ids or [b[0] for b in BUOYS]
    summaries = []
    for buoy_id, _label, lat, lon, _scope in BUOYS:
        if buoy_id not in targets:
            continue
        try:
            s = _backfill_buoy(buoy_id, lat, lon, cycles, force=force)
            summaries.append(s)
        except Exception:
            print(f"  {buoy_id}: FAILED", flush=True)
            traceback.print_exc()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "gee_backfill.log").open("a") as f:
        f.write(f"[{datetime.now(dtz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
                f"{summaries}\n")
    return summaries


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc2.gee_backfill")
    ap.add_argument("--buoy", default=None,
                    help="Limit to one buoy id (default: all 8 CSC2 buoys).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing per-cycle shards.")
    args = ap.parse_args()
    ids = [args.buoy] if args.buoy else None
    summaries = run_backfill(buoy_ids=ids, force=args.force)
    total_cycles = sum(s["cycles"] for s in summaries)
    total_rows = sum(s["rows"] for s in summaries)
    print(f"\n[gee_backfill] done: {total_cycles} cycles, {total_rows} rows "
          f"across {len(summaries)} buoys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
