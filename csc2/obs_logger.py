"""CSC2 observation logger — captures NDBC buoy spectral decompositions
every 30 minutes for the 8 CSC2 buoys.

Live rows are written to
    .csc_data/live_log/observations/buoy=<id>/year=Y/month=M/day=D.parquet

Historical NDBC stdmet yearly archives (written by csc2.ndbc_backfill)
live alongside under .csc_data/observations/buoy=<id>/year=Y/ — together
they form the buoy-side training-truth corpus. Readers (archive_status,
future trainers) combine both trees.

Schema (shared with stdmet historical archive):
    buoy_id, valid_utc, partition, hs_m, tp_s, dp_deg, source, ingest_utc

Runs as com.colesurfs.csc2-obs (every 1800s) independent of the dashboard.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone as dtz
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csc2.schema import BUOYS, LOGS_DIR, ensure_dirs  # noqa: E402
from buoy import fetch_buoy  # noqa: E402

OBS_DIR = _ROOT / ".csc_data" / "live_log" / "observations"

# Height stored in meters (SI) to match the stdmet historical archive schema
# (.csc_data/observations/) so archive readers don't need a per-source
# fallback column. Conversion lives here because buoy.fetch_buoy returns ft.
FT_PER_M = 3.28084


def _ft_to_m(v):
    try:
        return float(v) / FT_PER_M
    except (TypeError, ValueError):
        return None


def _daily_shard(buoy_id: str, valid_utc: datetime) -> Path:
    return (OBS_DIR / f"buoy={buoy_id}" / f"year={valid_utc.year}"
            / f"month={valid_utc.month:02d}"
            / f"day={valid_utc.day:02d}.parquet")


def _iso_z(t_str: str) -> str:
    """Normalize any ISO-8601 UTC timestamp (with `Z`, `+00:00`, microseconds,
    or naive) to the canonical `%Y-%m-%dT%H:%M:%SZ` form. Returns the input
    unchanged if parsing fails."""
    if not t_str:
        return t_str
    iso = t_str.replace("Z", "+00:00") if t_str.endswith("Z") else t_str
    try:
        t = datetime.fromisoformat(iso)
    except Exception:
        return t_str
    if t.tzinfo is None:
        t = t.replace(tzinfo=dtz.utc)
    return t.astimezone(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rows_for_buoy(buoy_id: str, now_utc: str) -> list[dict]:
    """Fetch a buoy via the main-site path and produce rows for every
    partition that came back. Uses the same buoy.fetch_buoy used by the
    dashboard — deduping happens at write time via valid_utc."""
    try:
        b = fetch_buoy(buoy_id)
    except Exception as e:
        print(f"  {buoy_id}: fetch failed — {type(e).__name__}: {e}")
        return []
    if not b or b.get("_offline"):
        return []

    valid_utc = _iso_z(b.get("time_utc") or b.get("obs_time_utc") or now_utc)
    rows: list[dict] = []
    # Partition 0 = combined / "now" reading (matches pre-CSC2 convention).
    # buoy.fetch_buoy returns wave_height_ft / wave_period_s / wave_direction_deg.
    rows.append({
        "buoy_id":    buoy_id,
        "valid_utc":  valid_utc,
        "partition":  0,
        "hs_m":       _ft_to_m(b.get("wave_height_ft")),
        "tp_s":       b.get("wave_period_s"),
        "dp_deg":     b.get("wave_direction_deg"),
        "source":     "ndbc_realtime",
        "ingest_utc": now_utc,
    })
    for i, c in enumerate(b.get("components") or [], start=1):
        rows.append({
            "buoy_id":    buoy_id,
            "valid_utc":  valid_utc,
            "partition":  i,
            "hs_m":       _ft_to_m(c.get("height_ft")),
            "tp_s":       c.get("period_s"),
            "dp_deg":     c.get("direction_deg"),
            "source":     "ndbc_realtime_spectral",
            "ingest_utc": now_utc,
        })
    return rows


def _append_dedup(buoy_id: str, new_rows: list[dict]) -> int:
    """Append new_rows to the daily shard, deduping on (valid_utc, partition).
    Later ingest_utc wins on conflict."""
    if not new_rows:
        return 0
    # Group by (year, month, day) of valid_utc
    by_day: dict[Path, list[dict]] = {}
    for r in new_rows:
        try:
            t = datetime.fromisoformat(r["valid_utc"].replace("Z", "+00:00"))
        except Exception:
            continue
        shard = _daily_shard(buoy_id, t.astimezone(dtz.utc))
        by_day.setdefault(shard, []).append(r)

    written = 0
    for shard, rows in by_day.items():
        shard.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(rows)
        new_df["valid_utc"] = new_df["valid_utc"].astype(str).map(_iso_z)
        old_df = None
        if shard.exists():
            try:
                old_df = pd.read_parquet(shard)
            except Exception:
                old_df = None

        # Fast path: if every (valid_utc, partition) pair from this tick is
        # already present in the shard, skip the rewrite entirely. NDBC
        # stdmet updates at 10-min intervals while the logger polls every
        # 30 min, so most ticks produce duplicates.
        if old_df is not None and len(old_df) > 0:
            old_keys = set(zip(old_df["valid_utc"].astype(str).map(_iso_z),
                                old_df["partition"]))
            new_keys = set(zip(new_df["valid_utc"],
                                new_df["partition"]))
            if new_keys.issubset(old_keys):
                continue

        if old_df is not None:
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        # Re-normalize valid_utc so legacy `+00:00` rows converge with new
        # `Z`-suffix writes — dedup is a string comparison.
        combined["valid_utc"] = combined["valid_utc"].astype(str).map(_iso_z)
        # Dedup: keep latest ingest_utc per (valid_utc, partition)
        combined = (combined.sort_values("ingest_utc")
                             .drop_duplicates(subset=["valid_utc", "partition"],
                                               keep="last"))
        combined.to_parquet(shard, index=False, compression="snappy")
        written += len(new_df)
    return written


def run_once() -> dict:
    ensure_dirs()
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    per_buoy: dict[str, int] = {}
    errors = 0
    for buoy_id, label, *_ in BUOYS:
        rows = _rows_for_buoy(buoy_id, now_utc)
        if not rows:
            errors += 1
            per_buoy[buoy_id] = 0
            continue
        n = _append_dedup(buoy_id, rows)
        per_buoy[buoy_id] = n
    elapsed = time.monotonic() - t0
    summary = {
        "when":     now_utc,
        "elapsed_s": round(elapsed, 1),
        "per_buoy": per_buoy,
        "errors":   errors,
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "obs_logger.log").open("a") as f:
        f.write(f"[{now_utc}] elapsed={elapsed:.1f}s  "
                f"buoys_ok={sum(1 for v in per_buoy.values() if v)} "
                f"rows={sum(per_buoy.values())} errors={errors}\n")
    return summary


def main() -> int:
    s = run_once()
    total = sum(s["per_buoy"].values())
    print(f"[csc2.obs_logger] {s['when']} elapsed={s['elapsed_s']}s "
          f"total_rows={total} errors={s['errors']}")
    for bid, n in s["per_buoy"].items():
        print(f"  {bid}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
