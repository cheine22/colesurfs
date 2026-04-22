"""CSC2 forecast logger.

Persists the exact dashboard-comparable swell-partition forecasts at each
CSC2 buoy to disk, one parquet per (model × buoy × cycle). Runs on a
launchd schedule (04Z and 16Z — roughly 4 h after CMEMS ANFC's 00Z/12Z
cycles publish) and captures the full +0..+240 h forecast trajectory.

Why we log live rather than backfill: CMEMS ANFC exposes only the current
forecast cycle (past cycles are overwritten), and no free third-party
archive preserves ECMWF-WAM swell-partition forecasts with lead-time
structure. The only path to a training corpus for a forecast-correction
model at these buoys is to start collecting now and let the archive grow.

The logger calls waves_cmems and waves with the same parameters the main
dashboard uses, so each logged row is byte-identical to what the user sees.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Adjust sys.path so a plain `python csc2/logger.py` run works from cron/launchd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csc2.schema import (  # noqa: E402
    BUOYS, FORECAST_COLUMNS, FORECASTS_DIR, LOGS_DIR, ensure_dirs,
)
from waves_cmems import fetch_cmems_point  # noqa: E402
from waves import fetch_wave_forecast  # noqa: E402

TIMEZONE = "America/New_York"


def _cycle_id(now_utc: datetime) -> str:
    """Rough CMEMS cycle anchor — last 00Z or 12Z before `now_utc`.

    CMEMS ANFC publishes at 00Z and 12Z; the logger is scheduled ~4 h later.
    We tag each cycle with the nominal run hour rather than fetch time."""
    h = 0 if now_utc.hour < 12 else 12
    return now_utc.replace(hour=h, minute=0, second=0, microsecond=0
                           ).strftime("%Y%m%dT%HZ")


def _local_to_utc(time_str: str) -> str:
    """Convert an America/New_York ISO timestamp (as produced by waves /
    waves_cmems) to a UTC ISO-8601 string. Returns '' on parse failure."""
    try:
        from zoneinfo import ZoneInfo
        t_local = datetime.fromisoformat(time_str).replace(tzinfo=ZoneInfo(TIMEZONE))
        return t_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def records_to_rows(records: list[dict], *, buoy_id: str, model: str,
                     cycle_utc: str, ingest_utc: str) -> list[dict]:
    """Flatten dashboard-format records into FORECAST_COLUMNS rows."""
    rows = []
    for r in records:
        valid_utc = _local_to_utc(r["time"])
        if not valid_utc:
            continue
        # lead_hours = valid_utc - cycle_utc (integer hours)
        try:
            vdt = datetime.strptime(valid_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            cdt = datetime.strptime(cycle_utc, "%Y%m%dT%HZ").replace(tzinfo=timezone.utc)
            lead_hours = int(round((vdt - cdt).total_seconds() / 3600))
        except Exception:
            lead_hours = None
        comps = r.get("components") or []
        c0 = comps[0] if len(comps) > 0 else {}
        c1 = comps[1] if len(comps) > 1 else {}
        rows.append({
            "buoy_id":             buoy_id,
            "model":               model,
            "cycle_utc":           cycle_utc,
            "valid_utc":           valid_utc,
            "lead_hours":          lead_hours,
            "sw1_height_ft":       c0.get("height_ft"),
            "sw1_period_s":        c0.get("period_s"),
            "sw1_direction_deg":   c0.get("direction_deg"),
            "sw2_height_ft":       c1.get("height_ft"),
            "sw2_period_s":        c1.get("period_s"),
            "sw2_direction_deg":   c1.get("direction_deg"),
            "combined_height_m":   r.get("combined_wave_height_m"),
            "combined_period_s":   r.get("combined_wave_period_s"),
            "combined_direction_deg": r.get("combined_wave_direction_deg"),
            "ingest_utc":          ingest_utc,
        })
    return rows


def shard_path(buoy_id: str, model: str, cycle_utc: str) -> Path:
    """Parquet path for one (buoy, model, cycle) triple."""
    year = cycle_utc[:4]
    month = cycle_utc[4:6]
    return (FORECASTS_DIR / f"model={model}" / f"buoy={buoy_id}"
            / f"year={year}" / f"month={month}"
            / f"cycle={cycle_utc}.parquet")


# Archive-status cache invalidates quickly when any writer touches this
# file — see csc2.archive_status._cache_is_stale.
FORECASTS_SENTINEL = FORECASTS_DIR / ".last_write"


def write_rows(buoy_id: str, model: str, cycle_utc: str,
                rows: list[dict]) -> int:
    """Write one cycle's rows to parquet and bump the archive-status
    sentinel so the next /api/csc2/archive_status hit recomputes."""
    if not rows:
        return 0
    path = shard_path(buoy_id, model, cycle_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    df.to_parquet(path, index=False, compression="snappy")
    # Touch the sentinel so archive_status knows something changed.
    try:
        FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        FORECASTS_SENTINEL.touch()
    except Exception:
        pass
    return len(df)


# Backward-compatible aliases. Any caller that already imported the
# underscored names (e.g. a long-running subprocess spawned before this
# rename) keeps working.
_shard_path      = shard_path
_write_rows      = write_rows
_records_to_rows = records_to_rows


def log_cycle(*, force: bool = False) -> dict:
    """Fetch and persist one cycle for all 8 buoys × {EURO, GFS}. Returns a
    short summary dict. `force=True` rewrites even if the cycle shard exists."""
    ensure_dirs()
    now_utc = datetime.now(timezone.utc)
    cycle_utc = _cycle_id(now_utc)
    ingest_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()

    coverage = {"EURO": {}, "GFS": {}}
    errors: list[str] = []

    for buoy_id, _label, lat, lon, _scope in BUOYS:
        # EURO (CMEMS)
        path_e = _shard_path(buoy_id, "EURO", cycle_utc)
        if path_e.exists() and not force:
            try:
                coverage["EURO"][buoy_id] = len(pd.read_parquet(path_e))
            except Exception:
                coverage["EURO"][buoy_id] = 0
        else:
            try:
                recs = fetch_cmems_point(lat, lon)
            except Exception as e:
                errors.append(f"EURO/{buoy_id}: {type(e).__name__}: {e}")
                recs = None
            rows = records_to_rows(recs or [], buoy_id=buoy_id, model="EURO",
                                     cycle_utc=cycle_utc, ingest_utc=ingest_utc)
            coverage["EURO"][buoy_id] = _write_rows(buoy_id, "EURO", cycle_utc, rows)

        # GFS (Open-Meteo ncep_gfswave025)
        path_g = _shard_path(buoy_id, "GFS", cycle_utc)
        if path_g.exists() and not force:
            try:
                coverage["GFS"][buoy_id] = len(pd.read_parquet(path_g))
            except Exception:
                coverage["GFS"][buoy_id] = 0
        else:
            try:
                recs = fetch_wave_forecast(lat, lon, "GFS")
            except Exception as e:
                errors.append(f"GFS/{buoy_id}: {type(e).__name__}: {e}")
                recs = None
            rows = records_to_rows(recs or [], buoy_id=buoy_id, model="GFS",
                                     cycle_utc=cycle_utc, ingest_utc=ingest_utc)
            coverage["GFS"][buoy_id] = _write_rows(buoy_id, "GFS", cycle_utc, rows)

    elapsed = time.monotonic() - t0
    summary = {
        "cycle_utc": cycle_utc,
        "elapsed_s": round(elapsed, 1),
        "coverage": coverage,
        "errors": errors,
    }
    _append_log(summary)
    return summary


def _append_log(summary: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = (
        f"[{summary['cycle_utc']}] elapsed={summary['elapsed_s']}s  "
        f"EURO rows/buoy={sum(summary['coverage']['EURO'].values())}  "
        f"GFS rows/buoy={sum(summary['coverage']['GFS'].values())}  "
        f"errors={len(summary['errors'])}\n"
    )
    with (LOGS_DIR / "logger.log").open("a") as f:
        f.write(line)
        for err in summary["errors"]:
            f.write(f"    ERR {err}\n")


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc2.logger")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing cycle shards.")
    args = ap.parse_args()
    try:
        s = log_cycle(force=args.force)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"[csc2.logger] cycle={s['cycle_utc']} elapsed={s['elapsed_s']}s")
    for m in ("EURO", "GFS"):
        by = s["coverage"][m]
        total = sum(by.values())
        print(f"  {m:<4}: {total} rows across {len(by)} buoys")
    if s["errors"]:
        print(f"  {len(s['errors'])} error(s):")
        for e in s["errors"]:
            print(f"    - {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
