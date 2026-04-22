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


# Calendar-day baselines — one cycle/day is the archive's effective density
# (GEE's CMEMS mirror ingests only the 00Z run). The live logger will catch
# both 00Z and 12Z going forward; once we cross ~12 months of live data the
# effective rate rises to ~1.5 cycles/day, which the progress bars can be
# re-tuned against if we want to represent sample density rather than
# calendar coverage. For now they represent calendar coverage.
SOFT_FLOOR_CYCLES = 90    # ~3 months at 1 cycle/day
TARGET_CYCLES     = 730   # ~24 months at 1 cycle/day

# Minimum fraction of the shared EURO∩GFS forecast window that must have a
# matching buoy observation for the cycle to count as paired. 1.00 requires
# perfect coverage (buoys miss individual hours routinely, so this is too
# strict); 0.95 tolerates a handful of obs dropouts per cycle while still
# guaranteeing training data is substantially complete.
BUOY_COVERAGE_THRESHOLD = 0.95
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
    """Training-ready buoy observation valid_utcs, hour-snapped.

    A row counts iff it carries complete combined-sea data (Hs, Tp,
    direction all non-null) on partition 0 — i.e. iff it can actually be
    consumed as a training target by the correction model. Raw NDBC
    stdmet timestamps land at :10/:26/:40/:56 past the hour depending on
    the buoy's sample schedule, so we snap to the nearest hour UTC before
    intersecting with forecast valid_utcs (which are on the hour).
    """
    acc: set[str] = set()

    def _snap_to_hour(s: str) -> str | None:
        s = str(s).strip()
        if not s:
            return None
        iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
        try:
            t = datetime.fromisoformat(iso)
        except Exception:
            return None
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        t_utc = t.astimezone(timezone.utc)
        # Round to nearest hour (>= :30 rounds up)
        if t_utc.minute >= 30 or (t_utc.minute == 30 and t_utc.second > 0):
            from datetime import timedelta
            t_utc = t_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            t_utc = t_utc.replace(minute=0, second=0, microsecond=0)
        return t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _cols_for(shard: Path) -> list[str] | None:
        # The two on-disk schemas share partition/tp_s/dp_deg but differ on
        # the Hs column (stdmet uses hs_m; live uses hs_ft).
        try:
            import pyarrow.parquet as pq
            names = set(pq.ParquetFile(shard).schema.names)
        except Exception:
            return None
        hs_col = "hs_ft" if "hs_ft" in names else ("hs_m" if "hs_m" in names else None)
        if hs_col is None:
            return None
        need = [hs_col, "tp_s", "dp_deg", "partition", "valid_utc"]
        if not all(c in names for c in need):
            return None
        return need

    for root in (OBS_HIST_DIR, OBS_LIVE_DIR):
        p = root / f"buoy={buoy_id}"
        if not p.exists():
            continue
        for shard in p.rglob("*.parquet"):
            cols = _cols_for(shard)
            if cols is None:
                continue
            try:
                df = pd.read_parquet(shard, columns=cols)
            except Exception:
                continue
            hs_col = cols[0]
            df = df[df["partition"] == 0]
            df = df.dropna(subset=[hs_col, "tp_s", "dp_deg", "valid_utc"])
            for v in df["valid_utc"].astype(str).tolist():
                snapped = _snap_to_hour(v)
                if snapped:
                    acc.add(snapped)
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


def _doy_hist(cycle_ids: set[str], years_seen: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for cyc in cycle_ids:
        key = f"{cyc[4:6]}-{cyc[6:8]}"
        out[key] = out.get(key, 0) + 1
        years_seen.add(cyc[:4])
    return out


def _timeline(d: dict[str, int], field: str) -> list[dict]:
    return [{"doy": k, field: n} for k, n in sorted(d.items())]


def _compute() -> dict:
    by_buoy: dict[str, dict] = {}
    histograms: dict[str, dict] = {}       # key → {paired_by_doy, euro_by_doy, gfs_by_doy}
    years_seen: set[str] = set()

    # Combined-east accumulators — each (buoy × cycle) contributes +1 to the
    # DOY bucket, so five east buoys all holding the same cycle yields 5.
    east_euro_hist:   dict[str, int] = {}
    east_gfs_hist:    dict[str, int] = {}
    east_paired_hist: dict[str, int] = {}

    def _bump(dest: dict[str, int], cycle_ids) -> None:
        for cyc in cycle_ids:
            key = f"{cyc[4:6]}-{cyc[6:8]}"
            dest[key] = dest.get(key, 0) + 1
            years_seen.add(cyc[:4])

    for buoy_id, label, _lat, _lon, scope in BUOYS:
        euro_per_cycle = _per_cycle_valid_utcs("EURO", buoy_id)
        gfs_per_cycle  = _per_cycle_valid_utcs("GFS",  buoy_id)
        obs_valids = _obs_valid_utcs(buoy_id)

        euro_ids = set(euro_per_cycle)
        gfs_ids  = set(gfs_per_cycle)
        sorted_euro = sorted(euro_ids)
        sorted_gfs  = sorted(gfs_ids)

        # Per-buoy paired cycle set — passes 95% buoy-obs coverage on the
        # EURO∩GFS forecast window.
        buoy_paired: set[str] = set()
        paired_samples = 0
        for cyc in sorted(euro_ids & gfs_ids):
            shared_forecast = euro_per_cycle[cyc] & gfs_per_cycle[cyc]
            if not shared_forecast:
                continue
            matched = shared_forecast & obs_valids
            if len(matched) / len(shared_forecast) >= BUOY_COVERAGE_THRESHOLD:
                buoy_paired.add(cyc)
                paired_samples += len(matched)

        by_buoy[buoy_id] = {
            "label": label,
            "scope": scope,
            "euro_cycles":   len(euro_ids),
            "gfs_cycles":    len(gfs_ids),
            "paired_cycles": len(buoy_paired),
            "paired_samples": paired_samples,
            "progress_target": round(min(1.0, len(buoy_paired) / TARGET_CYCLES), 3),
            "earliest":   (sorted_euro[0] if sorted_euro else (sorted_gfs[0] if sorted_gfs else None)),
            "latest":     (sorted_euro[-1] if sorted_euro else (sorted_gfs[-1] if sorted_gfs else None)),
        }

        # Per-buoy DOY histograms.
        histograms[buoy_id] = {
            "paired_by_doy": _timeline(_doy_hist(buoy_paired, years_seen), "paired_runs"),
            "euro_by_doy":   _timeline(_doy_hist(euro_ids,   years_seen), "runs"),
            "gfs_by_doy":    _timeline(_doy_hist(gfs_ids,    years_seen), "runs"),
        }

        # East-coast combined view: each east buoy's cycles stack on top of
        # the others. If all five have a run on 2025-06-15, the bar = 5.
        if scope == "east":
            _bump(east_euro_hist,   euro_ids)
            _bump(east_gfs_hist,    gfs_ids)
            _bump(east_paired_hist, buoy_paired)

    total_paired = sum(v["paired_cycles"] for v in by_buoy.values()) / max(1, len(by_buoy))

    histograms["combined_east"] = {
        "paired_by_doy": _timeline(east_paired_hist, "paired_runs"),
        "euro_by_doy":   _timeline(east_euro_hist,   "runs"),
        "gfs_by_doy":    _timeline(east_gfs_hist,    "runs"),
    }

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_cycles": TARGET_CYCLES,
        "buoy_coverage_threshold": BUOY_COVERAGE_THRESHOLD,
        "mean_paired_cycles": round(total_paired, 1),
        "per_buoy": by_buoy,
        "histograms": histograms,
        "years_seen": sorted(years_seen),
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
