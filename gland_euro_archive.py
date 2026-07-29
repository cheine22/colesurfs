"""
colesurfs — G-Land ECMWF-WAM rolling archive

Keeps a rolling 14-day window of CMEMS ECMWF-WAM swell at G-Land's offshore
node so /gland's history mode can show EURO alongside GFS.

Why an archive at all: Open-Meteo serves its own past analysis, so GFS history
is a plain API call. CMEMS is different — `waves_cmems.fetch_cmems_point` is
pinned to the current forecast window (and the main dashboard depends on that
behaviour), and a cold CMEMS fetch takes ~90 s against an upstream that is
hard-capped at ~2 publish cycles a day. So we persist what we have seen.

Nothing here mutates shared state. The processing path is the one CLAUDE.md
names as canonical for historical EURO — `waves_cmems.raw_rows_to_hourly_records`
— so archived rows are byte-identical to what the live dashboard would render
for the same hour.

Usage:
    python -m gland_euro_archive            # top up, prune, report
    python -m gland_euro_archive --backfill # also pull the full 14-day window
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gland_data")
ARCHIVE_PATH = os.path.join(ARCHIVE_DIR, "euro_archive.json")
WINDOW_DAYS = 14


def _load() -> dict:
    try:
        with open(ARCHIVE_PATH) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(rows: dict) -> None:
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    tmp = ARCHIVE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, separators=(",", ":"))
    os.replace(tmp, ARCHIVE_PATH)      # atomic: a reader never sees a half file


def _prune(rows: dict) -> dict:
    """Drop anything older than the rolling window."""
    import gland
    cutoff = (datetime.now(timezone.utc) + timedelta(hours=7)
              - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M")
    return {k: v for k, v in rows.items() if k >= cutoff}


def _fetch_window(start_utc: datetime, end_utc: datetime):
    """CMEMS EURO at the G-Land offshore node over an explicit UTC window.

    Deliberately NOT via waves_cmems.fetch_cmems_point — that is pinned to the
    forecast window and shared with the dashboard. We open the same product
    with our own window and hand the raw rows to the canonical processor, so
    the output matches the live path exactly.
    """
    import gland
    from waves_cmems import (CMEMS_PRODUCT, CMEMS_VARS, _extract_point_rows,
                            raw_rows_to_hourly_records)
    try:
        import copernicusmarine as cm
    except ImportError:
        print("[gland-euro] copernicusmarine not installed", file=sys.stderr)
        return None

    lat, lon = gland.SWELL_NODE_LAT, gland.SWELL_NODE_LON
    pad = 0.12
    try:
        ds = cm.open_dataset(
            dataset_id=CMEMS_PRODUCT, variables=CMEMS_VARS,
            minimum_longitude=lon - pad, maximum_longitude=lon + pad,
            minimum_latitude=lat - pad, maximum_latitude=lat + pad,
            start_datetime=start_utc.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        )
    except Exception as e:
        print(f"[gland-euro] open_dataset {type(e).__name__}: {e}", file=sys.stderr)
        return None
    try:
        raw = _extract_point_rows(ds, lat, lon)
        return raw_rows_to_hourly_records(raw)
    except Exception as e:
        print(f"[gland-euro] processing {type(e).__name__}: {e}", file=sys.stderr)
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def update(backfill: bool = False) -> dict:
    """Top up the archive and prune it to the rolling window."""
    import gland
    rows = _load()
    before = len(rows)

    now = datetime.now(timezone.utc)
    if backfill or not rows:
        start, end = now - timedelta(days=WINDOW_DAYS), now
        print(f"[gland-euro] backfilling {WINDOW_DAYS} d")
    else:
        # Normal top-up: re-pull a couple of days so late-arriving analysis
        # steps overwrite anything provisional.
        start, end = now - timedelta(days=2), now
    recs = _fetch_window(start, end)
    if not recs:
        print("[gland-euro] no records fetched")
        return {"added": 0, "total": before, "ok": False}

    added = 0
    for r in recs:
        t = r.get("time")
        if not t:
            continue
        best, scored = gland.pick_gland_swell(r.get("components") or [])
        entry = {"time": t, "model": "EURO", "components": scored,
                 "gland_swell": best}
        if t not in rows:
            added += 1
        rows[t] = entry

    rows = _prune(rows)
    _save(rows)
    span = (min(rows), max(rows)) if rows else (None, None)
    print(f"[gland-euro] +{added} new, {len(rows)} rows, {span[0]} .. {span[1]}")
    return {"added": added, "total": len(rows), "ok": True,
            "first": span[0], "last": span[1]}


def load_archive_rows() -> list:
    """Archived EURO rows, oldest first, in fetch_euro_waves() shape."""
    rows = _load()
    return [rows[k] for k in sorted(rows)]


def archive_status() -> dict:
    rows = _load()
    if not rows:
        return {"rows": 0, "first": None, "last": None, "days": 0}
    ks = sorted(rows)
    try:
        d0 = datetime.fromisoformat(ks[0]); d1 = datetime.fromisoformat(ks[-1])
        days = round((d1 - d0).total_seconds() / 86400, 1)
    except ValueError:
        days = 0
    return {"rows": len(rows), "first": ks[0], "last": ks[-1], "days": days}


if __name__ == "__main__":
    sys.exit(0 if update("--backfill" in sys.argv).get("ok") else 1)
