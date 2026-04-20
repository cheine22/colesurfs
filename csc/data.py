"""CSC data layer — read Parquet archives and build the training frame.

Two canonical readers:
  read_forecasts(lead_days=0)     → long DataFrame of model analysis values
  read_observations()             → long DataFrame of NDBC buoy values

One canonical joiner:
  build_training_frame()          → wide DataFrame, one row per
                                    (buoy, valid_utc), with columns for
                                    each (model, variable) analysis value
                                    and matched observation.

All timestamps land as pandas datetime64[ns, UTC].
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from csc.schema import FORECASTS_DIR, LIVE_LOG_DIR, OBSERVATIONS_DIR


# ─── Readers ──────────────────────────────────────────────────────────────

def _read_all_parquets(root: Path, glob: str) -> pd.DataFrame:
    """Read every Parquet file under `root` matching `glob`, concatenated.
    Returns an empty frame if nothing matches."""
    paths = sorted(root.rglob(glob))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def read_forecasts(lead_days: int | Iterable[int] = 0,
                   include_live_log: bool = True) -> pd.DataFrame:
    """Load model forecast rows from the archive.

    Args:
      lead_days: int or iterable. 0 = analysis only (default). Pass a list
        like [0, 1, 2, ..., 7] to include the previous_day rolling-window
        leads from the live logger.
      include_live_log: also read .csc_data/live_log/forecasts/ (not just
        the backfill under .csc_data/forecasts/).
    """
    if isinstance(lead_days, int):
        wanted = {lead_days}
    else:
        wanted = set(lead_days)

    frames = []
    back = _read_all_parquets(FORECASTS_DIR, "*.parquet")
    if not back.empty:
        frames.append(back)
    if include_live_log:
        live = _read_all_parquets(LIVE_LOG_DIR / "forecasts", "*.parquet")
        if not live.empty:
            frames.append(live)
    if not frames:
        return pd.DataFrame(columns=[
            "buoy_id", "model", "valid_utc", "lead_days",
            "variable", "value", "source", "ingest_utc",
        ])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["lead_days"].isin(wanted)]
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True)
    df = df.sort_values(["buoy_id", "model", "valid_utc", "lead_days", "variable"])
    df = df.drop_duplicates(
        subset=["buoy_id", "model", "valid_utc", "lead_days", "variable"],
        keep="last",   # later ingest wins
    )
    return df.reset_index(drop=True)


def read_observations(include_live_log: bool = True) -> pd.DataFrame:
    """Load NDBC observations from the archive.

    Columns: buoy_id, valid_utc, partition, hs_m, tp_s, dp_deg, source,
    ingest_utc. `partition=0` is combined WVHT (not a swell partition).
    Backfill rows use hs_m/tp_s/dp_deg in SI units. Live-log rows land
    in hs_ft/tp_s/dp_deg — we normalize here to SI (hs_m).
    """
    frames = []
    back = _read_all_parquets(OBSERVATIONS_DIR, "*.parquet")
    if not back.empty:
        frames.append(back)
    if include_live_log:
        live = _read_all_parquets(LIVE_LOG_DIR / "observations", "*.parquet")
        if not live.empty:
            # live-log rows use hs_ft — convert to hs_m for a uniform schema
            if "hs_ft" in live.columns and "hs_m" not in live.columns:
                live = live.copy()
                live["hs_m"] = live["hs_ft"] / 3.28084
                live = live.drop(columns=["hs_ft"])
            elif "hs_ft" in live.columns and "hs_m" in live.columns:
                # Prefer hs_m if present; coalesce
                live = live.copy()
                live["hs_m"] = live["hs_m"].fillna(live["hs_ft"] / 3.28084)
                live = live.drop(columns=["hs_ft"])
            frames.append(live)
    if not frames:
        return pd.DataFrame(columns=[
            "buoy_id", "valid_utc", "partition", "hs_m", "tp_s", "dp_deg",
            "source", "ingest_utc",
        ])
    df = pd.concat(frames, ignore_index=True)
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True)
    df = df.sort_values(["buoy_id", "valid_utc", "partition"])
    df = df.drop_duplicates(
        subset=["buoy_id", "valid_utc", "partition"],
        keep="last",
    )
    return df.reset_index(drop=True)


# ─── Wide-frame join for training ─────────────────────────────────────────

def _wide_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Pivot long forecast rows → wide, one row per (buoy, valid_utc, lead_days)
    with columns like gfs_wave_height, euro_swell_wave_period, etc."""
    if forecasts.empty:
        return forecasts
    df = forecasts.copy()
    df["colname"] = df["model"].str.lower() + "_" + df["variable"]
    wide = df.pivot_table(
        index=["buoy_id", "valid_utc", "lead_days"],
        columns="colname",
        values="value",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    return wide


def build_training_frame(
    lead_days: int = 0,
    tolerance: str = "30min",
    partition: int = 0,
) -> pd.DataFrame:
    """Join forecasts (at chosen lead) with observations (at chosen partition)
    using a ±tolerance nearest-time match.

    Returns a wide DataFrame with columns:
      buoy_id, valid_utc, lead_days,
      gfs_* (model analysis variables, one per OM_WAVE_VARS),
      euro_*,
      obs_hs_m, obs_tp_s, obs_dp_deg      ← matched NDBC observation
    """
    fc = read_forecasts(lead_days=lead_days)
    obs = read_observations()
    if fc.empty or obs.empty:
        return pd.DataFrame()

    wide = _wide_forecasts(fc)
    obs_part = obs[obs["partition"] == partition][
        ["buoy_id", "valid_utc", "hs_m", "tp_s", "dp_deg"]
    ].rename(columns={"hs_m": "obs_hs_m", "tp_s": "obs_tp_s", "dp_deg": "obs_dp_deg"})

    # merge_asof requires BOTH frames sorted by the time key globally
    # (within-group sorting is not enough; pandas checks monotonicity on
    # the whole frame even with `by=`).
    wide = wide.sort_values("valid_utc").reset_index(drop=True)
    obs_part = obs_part.sort_values("valid_utc").reset_index(drop=True)

    # merge_asof + `by` joins per-group on the nearest-time key
    joined = pd.merge_asof(
        wide,
        obs_part,
        on="valid_utc",
        by="buoy_id",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    # Drop rows where no observation was found within tolerance
    joined = joined.dropna(subset=["obs_hs_m"]).reset_index(drop=True)
    return joined


# ─── v2: primary-swell target frame ───────────────────────────────────────


def _read_primary_swell_archive() -> pd.DataFrame:
    """Read the historical primary-swell Parquet shards written by
    `csc.backfill_primary_swell`. Columns:
      buoy_id, valid_utc, partition, hm0_m, tm_s, dir_deg, energy, source.
    """
    from csc.schema import CSC_DATA_DIR
    root = CSC_DATA_DIR / "primary_swell"
    if not root.exists():
        return pd.DataFrame()
    return _read_all_parquets(root, "*.parquet")


def _live_log_primary_obs() -> pd.DataFrame:
    """Project the live-log observation shards onto the primary-swell
    schema (hm0_m, tm_s, dir_deg). Live-log rows use hs_ft/tp_s/dp_deg at
    partition=1 (primary) or partition=2 (secondary); we convert to meters
    and rename so it stacks directly with the historical archive.
    """
    from csc.schema import LIVE_LOG_DIR
    live = _read_all_parquets(LIVE_LOG_DIR / "observations", "*.parquet")
    if live.empty:
        return live
    live = live[live["partition"] >= 1].copy()
    if "hs_ft" in live.columns and "hm0_m" not in live.columns:
        live["hm0_m"] = live["hs_ft"] / 3.28084
    elif "hs_m" in live.columns and "hm0_m" not in live.columns:
        live["hm0_m"] = live["hs_m"]
    if "tp_s" in live.columns and "tm_s" not in live.columns:
        live["tm_s"] = live["tp_s"]
    if "dp_deg" in live.columns and "dir_deg" not in live.columns:
        live["dir_deg"] = live["dp_deg"]
    keep = ["buoy_id", "valid_utc", "partition", "hm0_m", "tm_s", "dir_deg"]
    # Tag source if missing so downstream can tell historical from live.
    if "source" not in live.columns:
        live["source"] = "ndbc_live_spectral"
    keep.append("source")
    return live[[c for c in keep if c in live.columns]]


def read_primary_swell_observations() -> pd.DataFrame:
    """Union of historical `.csc_data/primary_swell/` shards and the live-log
    partition≥1 rows, normalized to a single schema:
      buoy_id, valid_utc(utc datetime), partition(int), hm0_m, tm_s, dir_deg,
      source.
    """
    frames: list[pd.DataFrame] = []
    hist = _read_primary_swell_archive()
    if not hist.empty:
        frames.append(hist[[c for c in hist.columns if c in (
            "buoy_id", "valid_utc", "partition", "hm0_m", "tm_s", "dir_deg",
            "source")]])
    live = _live_log_primary_obs()
    if not live.empty:
        frames.append(live)
    if not frames:
        return pd.DataFrame(columns=[
            "buoy_id", "valid_utc", "partition",
            "hm0_m", "tm_s", "dir_deg", "source",
        ])
    df = pd.concat(frames, ignore_index=True)
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True)
    df["partition"] = df["partition"].astype(int)
    df["buoy_id"] = df["buoy_id"].astype(str)
    df = df.sort_values(["buoy_id", "valid_utc", "partition"])
    df = df.drop_duplicates(
        subset=["buoy_id", "valid_utc", "partition"], keep="last"
    )
    return df.reset_index(drop=True)


def build_training_frame_primary(
    lead_days: int = 0,
    tolerance: str = "30min",
    partition: int = 1,
) -> pd.DataFrame:
    """Same join shape as `build_training_frame()` but with the primary-
    swell (surfable-partition) NDBC observation as the target.

    Returns a wide DataFrame with columns identical in name to the v1
    frame — `obs_hs_m`, `obs_tp_s`, `obs_dp_deg` — so downstream feature
    engineering and models don't need to know about the target switch.
    The *values* are NDBC primary-swell Hm0 (meters), Tm (seconds),
    direction (degrees) from spectral decomposition.

    Args:
      partition: 1 = primary swell (default; the surfable component),
                 2 = secondary swell. Match the convention used by
                 `buoy._spectral_components` and `csc.logger.log_observation`.
    """
    fc = read_forecasts(lead_days=lead_days)
    obs = read_primary_swell_observations()
    if fc.empty or obs.empty:
        return pd.DataFrame()

    wide = _wide_forecasts(fc)
    obs_part = obs[obs["partition"] == partition][
        ["buoy_id", "valid_utc", "hm0_m", "tm_s", "dir_deg"]
    ].rename(columns={"hm0_m": "obs_hs_m", "tm_s": "obs_tp_s",
                      "dir_deg": "obs_dp_deg"})

    wide = wide.sort_values("valid_utc").reset_index(drop=True)
    obs_part = obs_part.sort_values("valid_utc").reset_index(drop=True)

    joined = pd.merge_asof(
        wide,
        obs_part,
        on="valid_utc",
        by="buoy_id",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    joined = joined.dropna(subset=["obs_hs_m"]).reset_index(drop=True)
    return joined
