"""CSC2 forecast logger.

Persists the exact dashboard-comparable swell-partition forecasts at each
CSC2 buoy to disk, one parquet per (model × buoy × cycle). Runs on a
launchd schedule (3 AM + 3 PM ET — after CMEMS ANFC's 00Z/12Z cycles
publish) and captures the full +0..+240 h forecast trajectory.

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
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Adjust sys.path so a plain `python csc2/logger.py` run works from cron/launchd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zoneinfo import ZoneInfo  # noqa: E402

from config import TIMEZONE  # noqa: E402
from csc2.schema import (  # noqa: E402
    BUOYS, FORECAST_COLUMNS, FORECASTS_DIR, LOGS_DIR, ensure_dirs,
)
from waves_cmems import fetch_cmems_point, CMEMS_PRODUCT  # noqa: E402
from waves import fetch_wave_forecast  # noqa: E402

_LOCAL_TZ = ZoneInfo(TIMEZONE)


def _cycle_id(now_utc: datetime) -> str:
    """GFS cycle anchor — last 00Z or 12Z before `now_utc`. Open-Meteo serves
    the 00Z run by ~05Z and the 12Z run by ~17Z, so at the logger's 07Z/19Z
    captures this is the run actually on the wire."""
    h = 0 if now_utc.hour < 12 else 12
    return now_utc.replace(hour=h, minute=0, second=0, microsecond=0
                           ).strftime("%Y%m%dT%HZ")


def euro_cycle_id(capture_utc: datetime, last_valid_utc: datetime) -> str:
    """CMEMS ANFC run behind a fetched EURO series, from the data not the clock.

    Two bulletins a day (file mtimes, 2026-09): the 00Z run lands ~08:30Z and
    reaches D+10 00Z (240 h); the 12Z run lands ~20:50Z and also reaches
    D+10 00Z (228 h). So the run DAY is the last valid time minus ten days,
    and the run HOUR is 12 once the 12Z bulletin is out (from ~20:40Z on the
    run day). Tagging by fetch clock — 07Z → "00Z", 19Z → "12Z" — labelled
    every live EURO cycle 12 h late until 2026-09-09 (shards relabelled)."""
    run_day = (last_valid_utc - timedelta(hours=234)).date()   # 240 h, tolerant of a few missing end steps
    twelve_out = datetime(run_day.year, run_day.month, run_day.day, 20, 30, tzinfo=timezone.utc)
    return f"{run_day:%Y%m%d}T{12 if capture_utc >= twelve_out else 0:02d}Z"


def euro_cycle_from_records(capture_utc: datetime, recs: list[dict],
                            bulletin: tuple[str, int] | None = None) -> str | None:
    """Cycle id for a fetched EURO series. `bulletin` = (R-date, run hour) of
    the newest CMEMS bulletin on the server (latest_euro_bulletin); it settles
    the run hour whenever it agrees with the data on the run day, otherwise
    the clock rule in euro_cycle_id decides."""
    last = max((_local_to_utc(r.get("time", "")) for r in recs), default="")
    if not last:
        return None
    last_dt = datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    cyc = euro_cycle_id(capture_utc, last_dt)
    if bulletin and bulletin[0] == cyc[:8]:
        cyc = f"{bulletin[0]}T{bulletin[1]:02d}Z"
    return cyc


_BULLETIN_RE = re.compile(r"mfwamglocep_\d{10}_R(\d{8})_(\d\d)H")


def latest_euro_bulletin(now_utc: datetime, timeout_s: float = 120) -> tuple[str, int] | None:
    """(R-date 'YYYYMMDD', run hour) of the newest bulletin listed for the
    CMEMS dataset — file names carry the release date and run hour
    (mfwamglocep_<valid>_R<date>_<00|12>H.nc). Today's date first, then
    yesterday's (the 00Z run is only listed from ~08:30Z). None on any
    failure or after `timeout_s`, so a slow listing can never stall a cycle."""
    out: dict = {}

    def _list():
        try:
            import copernicusmarine
            for d in (now_utc, now_utc - timedelta(days=1)):
                r = copernicusmarine.get(dataset_id=CMEMS_PRODUCT, filter=f"*R{d:%Y%m%d}*",
                                         dry_run=True, disable_progress_bar=True)
                found = {(m.group(1), int(m.group(2)))
                         for f in (getattr(r, "files", None) or [])
                         for m in [_BULLETIN_RE.search(str(getattr(f, "s3_url", f)))] if m}
                if found:
                    out["b"] = max(found)
                    return
        except Exception as e:
            out["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_list, daemon=True)
    t.start(); t.join(timeout_s)
    if "b" not in out:
        print(f"[csc2.logger] bulletin listing unavailable ({out.get('err', 'timeout')}) — clock rule",
              file=sys.stderr)
    return out.get("b")


_SAME_COLS = ["valid_utc", "sw1_height_ft", "sw1_period_s", "sw1_direction_deg",
              "sw2_height_ft", "sw2_period_s", "combined_height_m"]


def _latest_shard(buoy_id: str, model: str) -> Path | None:
    d = FORECASTS_DIR / f"model={model}" / f"buoy={buoy_id}"
    files = sorted(d.rglob("cycle=*.parquet")) if d.exists() else []
    return files[-1] if files else None


def _same_series(rows: list[dict], path: Path) -> bool:
    """True when `rows` is the series already stored at `path` (same window,
    same values) — a re-fetch that saw no new bulletin."""
    try:
        old = pd.read_parquet(path, columns=_SAME_COLS)
    except Exception:
        return False
    new = pd.DataFrame(rows, columns=FORECAST_COLUMNS)[_SAME_COLS]
    if len(old) != len(new) or old.valid_utc.min() != new.valid_utc.min() or old.valid_utc.max() != new.valid_utc.max():
        return False
    m = old.merge(new, on="valid_utc", suffixes=("_a", "_b"))
    return len(m) == len(old) and all((m[c + "_a"].fillna(-1) == m[c + "_b"].fillna(-1)).all()
                                      for c in _SAME_COLS[1:])


def _local_to_utc(time_str: str) -> str:
    """Convert an America/New_York ISO timestamp (as produced by waves /
    waves_cmems) to a UTC ISO-8601 string. Returns '' on parse failure."""
    try:
        t_local = datetime.fromisoformat(time_str).replace(tzinfo=_LOCAL_TZ)
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
    except Exception as e:
        print(f"[csc2.logger] sentinel touch failed ({FORECASTS_SENTINEL}): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
    return len(df)


def _check_archive_freshness(now_utc: datetime) -> None:
    """Warn when the sentinel says no writer has touched the forecast
    archive in > 24 h — a missed cycle would otherwise stay invisible
    until someone inspected coverage on /csc."""
    try:
        age_h = (now_utc.timestamp() - FORECASTS_SENTINEL.stat().st_mtime) / 3600
    except OSError:
        return
    if age_h > 24:
        print(f"[csc2.logger] WARNING: last archive write was {age_h:.0f}h ago "
              f"— a prior cycle likely failed silently", file=sys.stderr)


def log_cycle(*, force: bool = False) -> dict:
    """Fetch and persist one cycle for all 8 buoys × {EURO, GFS}. Returns a
    short summary dict. `force=True` rewrites even if the cycle shard exists."""
    ensure_dirs()
    now_utc = datetime.now(timezone.utc)
    _check_archive_freshness(now_utc)
    cycle_utc = _cycle_id(now_utc)
    ingest_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    bulletin = latest_euro_bulletin(now_utc)
    # EURO is fetched fresh, not through the dashboard's shared TTL cache: a
    # cached series can predate the newest bulletin and would be logged twice.
    fetch_euro_fresh = getattr(fetch_cmems_point, "__wrapped__", fetch_cmems_point)

    coverage = {"EURO": {}, "GFS": {}}
    errors: list[str] = []
    euro_cycles: set[str] = set()

    def _existing(path: Path) -> int | None:
        if path.exists() and not force:
            try:
                return len(pd.read_parquet(path))
            except Exception as e:
                # Corrupt shard (e.g. interrupted write) — refetch and rewrite
                # rather than masking it as already-logged.
                print(f"[csc2.logger] corrupt shard {path}: "
                      f"{type(e).__name__}: {e} — refetching", file=sys.stderr)
        return None

    def _log_one(buoy_id: str, model: str, fetch_fn, lat: float, lon: float) -> int:
        cyc = cycle_utc
        if model == "GFS":
            n = _existing(shard_path(buoy_id, model, cyc))
            if n is not None:
                return n
        try:
            recs = fetch_fn(lat, lon)
        except Exception as e:
            errors.append(f"{model}/{buoy_id}: {type(e).__name__}: {e}")
            recs = None
        if model == "EURO":
            # The run behind a CMEMS series is only knowable from the data
            # (+ the bulletin listing); never from the fetch clock.
            cyc = euro_cycle_from_records(now_utc, recs or [], bulletin) or cyc
            euro_cycles.add(cyc)
            n = _existing(shard_path(buoy_id, model, cyc))
            if n is not None:
                return n
        rows = records_to_rows(recs or [], buoy_id=buoy_id, model=model,
                                 cycle_utc=cyc, ingest_utc=ingest_utc)
        if model == "EURO" and rows:
            prev = _latest_shard(buoy_id, model)
            if prev is not None and _same_series(rows, prev):
                errors.append(f"EURO/{buoy_id}: series unchanged since {prev.stem[6:]} — no new bulletin, not written")
                return 0
        return write_rows(buoy_id, model, cyc, rows)

    for buoy_id, _label, lat, lon, _scope in BUOYS:
        coverage["EURO"][buoy_id] = _log_one(buoy_id, "EURO", fetch_euro_fresh, lat, lon)
        coverage["GFS"][buoy_id] = _log_one(
            buoy_id, "GFS", lambda la, lo: fetch_wave_forecast(la, lo, "GFS"), lat, lon)

    elapsed = time.monotonic() - t0
    summary = {
        "cycle_utc": cycle_utc,
        "euro_cycle_utc": ",".join(sorted(euro_cycles)) or None,
        "elapsed_s": round(elapsed, 1),
        "coverage": coverage,
        "errors": errors,
    }
    _append_log(summary)
    return summary


def _append_log(summary: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = (
        f"[{summary['cycle_utc']} euro={summary.get('euro_cycle_utc')}] elapsed={summary['elapsed_s']}s  "
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
    print(f"[csc2.logger] cycle={s['cycle_utc']} euro={s.get('euro_cycle_utc')} elapsed={s['elapsed_s']}s")
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
