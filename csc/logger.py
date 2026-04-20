"""CSC forward logger.

Two responsibilities:

  1. Forecast logger — for each (buoy × model), query the Open-Meteo Marine
     API with `past_days=7, forecast_days=1` AND all `<var>_previous_day1..7`
     companions. One call captures:
       * hourly analysis values for the last 7 days (valid-time indexed)
       * for each of those valid-times, the forecasts issued 1..7 days
         before = lead times of 24, 48, ..., 168 h.
     Writes to .csc_data/live_log/forecasts/ as Parquet shards.

  2. Observation logger — append a hook-compatible function the main app's
     buoy.fetch_buoy can call on every successful fetch, persisting the
     spectral-decomposed swell partitions to .csc_data/live_log/observations/.

Both outputs are append-only Parquet daily shards partitioned by
(model, buoy, year) for forecasts and (buoy, year) for observations. A
nightly compactor dedupes and rewrites; it will live in a sibling module.

Invocation:
    python -m csc.logger               # one full tick: every buoy × model
    python -m csc.logger --buoy 44097  # one buoy, both models
    python -m csc.logger --models-only # skip observations (logger tick)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from csc.schema import (
    BUOYS, LIVE_LOG_DIR, OM_MODELS, OM_WAVE_VARS, PREV_DAY_LEADS,
    ensure_dirs, om_var_columns,
)

OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"


# ─── Forecast logger ──────────────────────────────────────────────────────

def _pull_one(buoy_id: str, lat: float, lon: float, model_key: str,
              timeout_s: float = 45.0) -> dict[str, Any] | None:
    """Return the raw Open-Meteo Marine API response for one (buoy, model)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "models": OM_MODELS[model_key],
        "past_days": 7,
        "forecast_days": 1,
        "hourly": ",".join(om_var_columns()),
        "timezone": "UTC",
    }
    try:
        r = requests.get(OPEN_METEO_MARINE, params=params, timeout=timeout_s)
        if not r.ok:
            print(f"  [!] {buoy_id}/{model_key}: HTTP {r.status_code} "
                  f"{r.text[:200]}", file=sys.stderr)
            return None
        return r.json()
    except requests.RequestException as e:
        print(f"  [!] {buoy_id}/{model_key}: {e}", file=sys.stderr)
        return None


def _long_rows_from_response(buoy_id: str, model_key: str,
                             resp: dict[str, Any],
                             ingest_ts: datetime) -> list[dict[str, Any]]:
    """Flatten the hourly response into long rows, one per
    (valid_time × variable × lead_days). lead_days=0 is the analysis value."""
    hourly = resp.get("hourly") or {}
    times: list[str] = hourly.get("time") or []
    rows: list[dict[str, Any]] = []
    ingest_s = ingest_ts.isoformat()

    for var in OM_WAVE_VARS:
        analysis_vals = hourly.get(var)
        if not analysis_vals:
            continue
        for t, v in zip(times, analysis_vals):
            if v is None:
                continue
            rows.append({
                "buoy_id": buoy_id,
                "model": model_key,
                "valid_utc": t,
                "lead_days": 0,
                "variable": var,
                "value": float(v),
                "ingest_utc": ingest_s,
            })

        for n in PREV_DAY_LEADS:
            colname = f"{var}_previous_day{n}"
            vals = hourly.get(colname)
            if not vals:
                continue
            for t, v in zip(times, vals):
                if v is None:
                    continue
                rows.append({
                    "buoy_id": buoy_id,
                    "model": model_key,
                    "valid_utc": t,
                    "lead_days": n,
                    "variable": var,
                    "value": float(v),
                    "ingest_utc": ingest_s,
                })
    return rows


def _append_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Append long rows to a Parquet shard. The first write creates the
    file; subsequent writes concat-and-rewrite (cheap at our row volume,
    simpler than Parquet row-group append)."""
    if not rows:
        return
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            df = pd.concat([existing, new], ignore_index=True)
        except Exception as e:
            # If the existing shard is corrupt, rename it rather than lose new data.
            corrupt = path.with_suffix(".parquet.corrupt")
            print(f"  [!] existing shard unreadable ({e}); moving to {corrupt.name}",
                  file=sys.stderr)
            path.rename(corrupt)
            df = new
    else:
        df = new
    df.to_parquet(path, index=False, compression="snappy")


def log_forecast(buoy_id: str, lat: float, lon: float, model_key: str,
                 ingest_ts: datetime | None = None) -> int:
    """Run one Marine API pull for (buoy, model) and append rows to today's
    forecast shard. Returns row count appended."""
    ensure_dirs()
    ingest_ts = ingest_ts or datetime.now(timezone.utc)
    resp = _pull_one(buoy_id, lat, lon, model_key)
    if resp is None:
        return 0
    rows = _long_rows_from_response(buoy_id, model_key, resp, ingest_ts)
    if not rows:
        return 0
    year = ingest_ts.strftime("%Y")
    day = ingest_ts.strftime("%Y-%m-%d")
    shard = (LIVE_LOG_DIR / "forecasts" /
             f"model={model_key}" / f"buoy={buoy_id}" / f"year={year}" /
             f"shard-{day}.parquet")
    _append_parquet(rows, shard)
    return len(rows)


# ─── Observation logger — called from buoy.py on every live fetch ─────────

def log_observation(buoy_id: str, obs_result: dict[str, Any] | None,
                    components: list[dict[str, Any]] | None = None,
                    ingest_ts: datetime | None = None) -> int:
    """Append one observation row (analysis + spectral partitions if given)
    to today's observation shard. Designed to be called from buoy.fetch_buoy
    wrapped in try/except — must never raise on the hot path.

    Args:
      buoy_id: NDBC station id.
      obs_result: the dict buoy.fetch_buoy returns (has keys like wvht_ft,
                  dpd_s, mwd_deg, time_utc etc). If None, caller had no obs.
      components: optional list of {hm0_ft, tm_s, dir_deg, ...} partitions
                  from the spectral decomposition.
      ingest_ts: override ingest timestamp (tests).

    Returns number of rows appended.
    """
    if obs_result is None or obs_result.get("_offline"):
        return 0
    ensure_dirs()
    ingest_ts = ingest_ts or datetime.now(timezone.utc)
    valid_utc = obs_result.get("timestamp")
    if valid_utc is None:
        return 0

    # Keys below match buoy.py's _parse() return and _spectral_components()
    # output — do not change without updating buoy.py in lockstep.
    rows: list[dict[str, Any]] = []
    wvht_ft = obs_result.get("wave_height_ft")
    dpd_s   = obs_result.get("wave_period_s")
    mwd_deg = obs_result.get("wave_direction_deg")
    if wvht_ft is not None or dpd_s is not None or mwd_deg is not None:
        rows.append({
            "buoy_id": buoy_id,
            "valid_utc": str(valid_utc),
            "partition": 0,                  # 0 = combined WVHT (not a swell partition)
            "hs_ft": float(wvht_ft) if wvht_ft is not None else None,
            "tp_s":  float(dpd_s) if dpd_s is not None else None,
            "dp_deg": float(mwd_deg) if mwd_deg is not None else None,
            "source": "ndbc_realtime",
            "ingest_utc": ingest_ts.isoformat(),
        })

    comps = components if components is not None else obs_result.get("components")
    if comps:
        for i, c in enumerate(comps, start=1):
            h = c.get("height_ft")
            p = c.get("period_s")
            d = c.get("direction_deg")
            if h is None and p is None and d is None:
                continue
            rows.append({
                "buoy_id": buoy_id,
                "valid_utc": str(valid_utc),
                "partition": i,
                "hs_ft": float(h) if h is not None else None,
                "tp_s":  float(p) if p is not None else None,
                "dp_deg": float(d) if d is not None else None,
                "source": "ndbc_realtime_spectral",
                "ingest_utc": ingest_ts.isoformat(),
            })

    if not rows:
        return 0

    year = ingest_ts.strftime("%Y")
    day = ingest_ts.strftime("%Y-%m-%d")
    shard = (LIVE_LOG_DIR / "observations" /
             f"buoy={buoy_id}" / f"year={year}" /
             f"shard-{day}.parquet")
    _append_parquet(rows, shard)
    return len(rows)


# ─── CLI tick ─────────────────────────────────────────────────────────────

def tick(only_buoy: str | None = None, sleep_s: float = 0.4) -> dict[str, int]:
    """Run one full logger tick: every (buoy × model) forecast pull.
    Observations are captured separately via the buoy.py hook — this tick
    is focused on forecast data."""
    ensure_dirs()
    ingest_ts = datetime.now(timezone.utc)
    stats = {"forecast_rows": 0, "model_pulls": 0, "errors": 0}
    for buoy_id, label, lat, lon, _op in BUOYS:
        if only_buoy and buoy_id != only_buoy:
            continue
        for model_key in OM_MODELS:
            try:
                n = log_forecast(buoy_id, lat, lon, model_key, ingest_ts)
                stats["forecast_rows"] += n
                stats["model_pulls"] += 1
                print(f"  {buoy_id:<6} {model_key:<5} → {n:>6} rows")
            except Exception:
                stats["errors"] += 1
                traceback.print_exc()
            time.sleep(sleep_s)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.logger")
    ap.add_argument("--buoy", default=None,
                    help="Limit to one buoy id (default: all).")
    args = ap.parse_args()

    start = time.time()
    stats = tick(only_buoy=args.buoy)
    elapsed = time.time() - start
    print(f"\nlogger tick: {stats['model_pulls']} pulls, "
          f"{stats['forecast_rows']} rows, {stats['errors']} errors, "
          f"{elapsed:.1f}s")
    return 0 if stats["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
