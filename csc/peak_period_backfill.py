"""Backfill the missing Open-Meteo variables into the existing forecast
archive so dashboardify has full inputs (critical for EURO where
`wave_peak_period` drives the wind-chop filter).

The main analysis archive under `.csc_data/forecasts/` was built at a
time when only 9 Open-Meteo variables were requested; the dashboard and
dashboardify now require 13 (+`wave_peak_period`, +`tertiary_swell_*`).
Rather than re-pull everything, this script:

  1. Reads the archive to find each (buoy, month) with data.
  2. Issues one Open-Meteo historical-forecast call per (buoy, month,
     model) covering only the missing variables.
  3. Appends the new long rows to the same parquet shard.
  4. Dedupes on (buoy, model, valid_utc, lead_days, variable).

Each monthly pull touches ~720 hourly rows × 4 vars = ~2880 rows.
Runtime ~10 min for the full East+West scope.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

from csc.schema import BUOYS, FORECASTS_DIR, OM_MODELS

HIST_API = "https://marine-api.open-meteo.com/v1/marine"

# Variables the dashboard uses that are missing from the 2024-2026
# legacy archive.
MISSING_VARS = [
    "wave_peak_period",
    "tertiary_swell_wave_height",
    "tertiary_swell_wave_period",
    "tertiary_swell_wave_direction",
]


def _month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Return list of (year, month) from start to end inclusive."""
    s_y, s_m = map(int, start.split("-")[:2])
    e_y, e_m = map(int, end.split("-")[:2])
    out = []
    y, m = s_y, s_m
    while (y, m) <= (e_y, e_m):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _pull_historical(lat: float, lon: float, model_key: str,
                     start_date: str, end_date: str,
                     variables: list[str]) -> dict | None:
    """One historical-forecast-api call for the missing variables."""
    params = {
        "latitude":  lat,
        "longitude": lon,
        "models":    OM_MODELS[model_key],
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":    ",".join(variables),
        "timezone":  "UTC",
    }
    for attempt in range(3):
        try:
            r = requests.get(HIST_API, params=params, timeout=90)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  [!] {model_key} {start_date}/{end_date}: {e}",
                  file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return None


def _response_to_rows(buoy_id: str, model_key: str, resp: dict,
                      variables: list[str]) -> list[dict]:
    hourly = resp.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return []
    ingest = datetime.now(timezone.utc).isoformat()
    rows = []
    for var in variables:
        vals = hourly.get(var)
        if not vals:
            continue
        for t, v in zip(times, vals):
            if v is None:
                continue
            rows.append({
                "buoy_id": buoy_id,
                "model":   model_key,
                "valid_utc": t,
                "lead_days": 0,
                "variable":  var,
                "value":    float(v),
                "ingest_utc": ingest,
            })
    return rows


def _shard_path(model_key: str, buoy_id: str, year: int, month: int) -> Path:
    return (FORECASTS_DIR
            / f"model={model_key}"
            / f"buoy={buoy_id}"
            / f"year={year}"
            / f"month={month:02d}"
            / "analysis.parquet")


def _append_and_dedupe(shard: Path, new_rows: list[dict]) -> int:
    """Append new rows to shard and dedupe by primary key, keeping latest."""
    if not new_rows:
        return 0
    shard.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(new_rows)
    if shard.exists():
        existing = pd.read_parquet(shard)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values("ingest_utc")
    combined = combined.drop_duplicates(
        subset=["buoy_id", "model", "valid_utc", "lead_days", "variable"],
        keep="last",
    )
    combined.to_parquet(shard, index=False, compression="snappy")
    return len(new_rows)


def run(start: str, end: str, only_buoy: str | None,
        only_model: str | None, sleep_s: float = 1.1) -> int:
    months = _month_range(start, end)
    total_rows = 0
    errors = 0

    for buoy_id, label, lat, lon, _op in BUOYS:
        if only_buoy and buoy_id != only_buoy:
            continue
        for model_key in OM_MODELS:
            if only_model and model_key != only_model:
                continue
            # EURO only exposes wave_peak_period (no partition vars) —
            # still need it for the chop filter.
            if model_key == "EURO":
                vars_to_pull = ["wave_peak_period"]
            else:
                vars_to_pull = MISSING_VARS
            for (y, m) in months:
                start_d = f"{y:04d}-{m:02d}-01"
                if m == 12:
                    end_d = f"{y + 1:04d}-01-01"
                else:
                    end_d = f"{y:04d}-{m + 1:02d}-01"
                # historical-forecast is inclusive; subtract a day
                end_incl = (pd.Timestamp(end_d) - pd.Timedelta(days=1)
                            ).strftime("%Y-%m-%d")

                shard = _shard_path(model_key, buoy_id, y, m)
                if not shard.exists():
                    # No existing data for this month — skip (the 9-var
                    # archive didn't cover it either).
                    continue

                # Check whether peak is already present in this shard.
                try:
                    existing = pd.read_parquet(shard, columns=["variable"])
                    present = set(existing["variable"].unique())
                except Exception:
                    present = set()
                missing_here = [v for v in vars_to_pull if v not in present]
                if not missing_here:
                    continue

                resp = _pull_historical(lat, lon, model_key, start_d,
                                        end_incl, missing_here)
                if resp is None:
                    errors += 1
                    time.sleep(sleep_s)
                    continue
                rows = _response_to_rows(buoy_id, model_key, resp,
                                         missing_here)
                added = _append_and_dedupe(shard, rows)
                total_rows += added
                print(f"  {model_key} {buoy_id} {y:04d}-{m:02d}  "
                      f"+{added} rows (vars={','.join(missing_here)})")
                time.sleep(sleep_s)

    print(f"\nDone — {total_rows} rows appended, {errors} errors.")
    return 0 if errors == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.peak_period_backfill")
    ap.add_argument("--start", default="2024-03",
                    help="Start month YYYY-MM")
    ap.add_argument("--end", default="2026-04",
                    help="End month YYYY-MM")
    ap.add_argument("--buoy", default=None)
    ap.add_argument("--model", default=None, choices=("GFS", "EURO"))
    ap.add_argument("--sleep", type=float, default=1.1)
    args = ap.parse_args()
    return run(args.start, args.end, args.buoy, args.model, args.sleep)


if __name__ == "__main__":
    sys.exit(main())
