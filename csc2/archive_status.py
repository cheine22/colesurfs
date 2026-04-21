"""How much CSC2 training data has been collected so far, per buoy.

Backs /api/csc2/archive_status. For each buoy reports:
  - EURO forecast cycles
  - GFS forecast cycles
  - paired cycles (min of EURO and GFS cycle counts)
  - buoy observations total, and observations that are PAIRED with forecasts
    (valid_utc appears in at least one EURO shard AND one GFS shard — these
    are the only obs that can be used as training targets)
  - earliest / latest dates
  - progress toward the 3-month soft floor and 24-month target

The paired-obs count is cached to `.csc2_data/archive_status_cache.json`
because it requires scanning every forecast shard. Cache is regenerated
when any forecast shard is newer than the cache file or the cache is
older than 30 minutes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from csc2.schema import BUOYS, CSC2_DATA_DIR, FORECASTS_DIR


SOFT_FLOOR_CYCLES = 180
TARGET_CYCLES = 1460
CACHE_PATH = CSC2_DATA_DIR / "archive_status_cache.json"
CACHE_TTL_SEC = 1800

# Historical obs archive (stdmet yearly) + live_log/observations (30-min live).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBS_HIST_DIR = PROJECT_ROOT / ".csc_data" / "observations"
OBS_LIVE_DIR = PROJECT_ROOT / ".csc_data" / "live_log" / "observations"


def _cycle_shards(model: str, buoy_id: str) -> list[Path]:
    root = FORECASTS_DIR / f"model={model}" / f"buoy={buoy_id}"
    return list(root.rglob("cycle=*.parquet")) if root.exists() else []


def _cycle_ids(shards: list[Path]) -> list[str]:
    out = []
    for s in shards:
        n = s.name
        if n.startswith("cycle=") and n.endswith(".parquet"):
            out.append(n[len("cycle="):-len(".parquet")])
    out.sort()
    return out


def _forecast_valid_utcs(model: str, buoy_id: str) -> set[str]:
    """Return the set of valid_utc strings this buoy has in any shard."""
    shards = _cycle_shards(model, buoy_id)
    if not shards:
        return set()
    acc: set[str] = set()
    for s in shards:
        try:
            df = pd.read_parquet(s, columns=["valid_utc"])
            acc.update(df["valid_utc"].dropna().astype(str).tolist())
        except Exception:
            continue
    return acc


def _per_cycle_valid_utcs(model: str, buoy_id: str) -> dict[str, set[str]]:
    """Map cycle_id → set of valid_utc strings for that cycle's shard."""
    out: dict[str, set[str]] = {}
    for s in _cycle_shards(model, buoy_id):
        n = s.name
        if not (n.startswith("cycle=") and n.endswith(".parquet")):
            continue
        cycle = n[len("cycle="):-len(".parquet")]
        try:
            df = pd.read_parquet(s, columns=["valid_utc"])
            out[cycle] = set(df["valid_utc"].dropna().astype(str).tolist())
        except Exception:
            out[cycle] = set()
    return out


def _obs_valid_utcs(buoy_id: str) -> set[str]:
    """Normalized valid_utc strings from historical stdmet + live obs.
    Strips trailing timezone, keeps ISO 'YYYY-MM-DDTHH:MM:SS' up to the 'Z'
    equivalent. Forecast shards store 'YYYY-MM-DDTHH:MM:SSZ' — we canonicalize
    everything to the same 'YYYY-MM-DDTHH:MM:SSZ' form."""
    acc: set[str] = set()

    def _norm(s: str) -> str | None:
        s = str(s).strip()
        if not s:
            return None
        if "+" in s:
            s = s.split("+", 1)[0]
        if s.endswith("Z"):
            return s
        return s + "Z"

    for root in (OBS_HIST_DIR, OBS_LIVE_DIR):
        p = root / f"buoy={buoy_id}"
        if not p.exists():
            continue
        for shard in p.rglob("*.parquet"):
            try:
                df = pd.read_parquet(shard, columns=["valid_utc"])
                for v in df["valid_utc"].dropna().astype(str).tolist():
                    n = _norm(v)
                    if n:
                        acc.add(n)
            except Exception:
                continue
    return acc


def _cache_is_stale() -> bool:
    if not CACHE_PATH.exists():
        return True
    mtime = CACHE_PATH.stat().st_mtime
    if datetime.now().timestamp() - mtime > CACHE_TTL_SEC:
        return True
    # Invalidate if any forecast shard is newer than the cache
    for model in ("EURO", "GFS"):
        root = FORECASTS_DIR / f"model={model}"
        if not root.exists():
            continue
        for shard in root.rglob("*.parquet"):
            if shard.stat().st_mtime > mtime:
                return True
    return False


def _compute() -> dict:
    by_buoy: dict[str, dict] = {}
    for buoy_id, label, _lat, _lon, scope in BUOYS:
        euro_per_cycle = _per_cycle_valid_utcs("EURO", buoy_id)
        gfs_per_cycle  = _per_cycle_valid_utcs("GFS",  buoy_id)
        obs_valids = _obs_valid_utcs(buoy_id)

        euro_ids = sorted(euro_per_cycle)
        gfs_ids  = sorted(gfs_per_cycle)

        # A cycle is "fully paired" iff EURO has a shard, GFS has a shard,
        # and the buoy has at least one observation whose valid_utc matches
        # a valid_utc present in BOTH forecasts for that cycle. This is the
        # minimum condition for deriving a trainable (forecast, forecast,
        # observation) triple from that init time.
        shared_cycles = sorted(set(euro_ids) & set(gfs_ids))
        paired_cycles = 0
        paired_samples = 0  # total lead-time samples (cycle × lead) with all three
        for cyc in shared_cycles:
            shared_valids = euro_per_cycle[cyc] & gfs_per_cycle[cyc] & obs_valids
            if shared_valids:
                paired_cycles += 1
                paired_samples += len(shared_valids)

        by_buoy[buoy_id] = {
            "label": label,
            "scope": scope,
            "euro_cycles":   len(euro_ids),
            "gfs_cycles":    len(gfs_ids),
            "paired_cycles": paired_cycles,
            "paired_samples": paired_samples,
            "progress_soft_floor": round(min(1.0, paired_cycles / SOFT_FLOOR_CYCLES), 3),
            "progress_target":     round(min(1.0, paired_cycles / TARGET_CYCLES), 3),
            "earliest":    (euro_ids[0] if euro_ids else (gfs_ids[0] if gfs_ids else None)),
            "latest":      (euro_ids[-1] if euro_ids else (gfs_ids[-1] if gfs_ids else None)),
        }
    total_paired = sum(v["paired_cycles"] for v in by_buoy.values()) / max(1, len(by_buoy))
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "soft_floor_cycles": SOFT_FLOOR_CYCLES,
        "target_cycles": TARGET_CYCLES,
        "mean_paired_cycles": round(total_paired, 1),
        "per_buoy": by_buoy,
    }


def summarize() -> dict:
    """Cached entry point used by /api/csc2/archive_status."""
    if not _cache_is_stale():
        try:
            with CACHE_PATH.open() as f:
                return json.load(f)
        except Exception:
            pass
    payload = _compute()
    try:
        CSC2_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w") as f:
            json.dump(payload, f)
    except Exception:
        pass
    return payload
