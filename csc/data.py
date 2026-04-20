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

from csc.schema import CSC_DATA_DIR, FORECASTS_DIR, LIVE_LOG_DIR, OBSERVATIONS_DIR


# ─── Readers ──────────────────────────────────────────────────────────────

def _read_all_parquets(root: Path, glob: str) -> pd.DataFrame:
    """Read every Parquet file under `root` matching `glob`, concatenated.
    Returns an empty frame if nothing matches."""
    paths = sorted(root.rglob(glob))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# Note: a CMEMS partitioned-WAM reader was prototyped here and reverted
# once it became clear CMEMS reports a different quantity (true primary-
# swell partition) than what the live dashboard renders for EURO (OM
# combined Hs filtered by wave_peak_period). Wiring CMEMS into training
# alone would reintroduce train/serve drift. Shards written by
# `csc/cmems_backfill.py` remain on disk under `.csc_data/euro_partitions/`
# as a reserved future source — they'd only make sense to consume if the
# dashboard itself switches to CMEMS for EURO rendering.


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

    # EURO quality gate — "if data does not meet quality, it cannot be used".
    # Open-Meteo EURO exposes no swell partitions, so dashboardify falls
    # back to combined Hs filtered by wave_peak_period ≥ 6s. That peak_period
    # variable only became available from Open-Meteo on 2025-11-01; before
    # that date, dashboardify's fallback uses mean period instead, which is
    # a DIFFERENT filter than what the live dashboard applies today.
    # Dropping those rows so training features are bit-identical to live
    # dashboard rendering for every hour we keep.
    peak_hours = df[
        (df["model"] == "EURO") & (df["variable"] == "wave_peak_period")
    ][["buoy_id", "valid_utc"]].drop_duplicates()
    if not peak_hours.empty:
        peak_keys = set(zip(
            peak_hours["buoy_id"].astype(str),
            peak_hours["valid_utc"].astype("int64"),
        ))
        euro_mask = df["model"] == "EURO"
        euro_rows = df[euro_mask]
        keep_euro_idx = euro_rows.index[
            [(b, t) in peak_keys for b, t in
             zip(euro_rows["buoy_id"].astype(str),
                 euro_rows["valid_utc"].astype("int64"))]
        ]
        drop_euro_idx = euro_rows.index.difference(keep_euro_idx)
        if len(drop_euro_idx):
            df = df.drop(index=drop_euro_idx)
    else:
        # No peak_period anywhere → all EURO rows fail the quality gate.
        df = df[df["model"] != "EURO"]

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

_RAW_OM_VARS = (
    "wave_height", "wave_period", "wave_peak_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "secondary_swell_wave_height", "secondary_swell_wave_period",
    "secondary_swell_wave_direction",
    "tertiary_swell_wave_height", "tertiary_swell_wave_period",
    "tertiary_swell_wave_direction",
)


def _wide_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Pivot long forecast rows → wide, then REPLACE the raw Open-Meteo
    columns with the exact values the live dashboard would display
    (`waves.py::_parse_response` output, via `csc.dashboardify`).

    Non-negotiable: training features at every (buoy, valid_utc) must
    equal what the dashboard shows for that (buoy, valid_utc). Feeding
    the ML layer raw Open-Meteo `wave_height` (combined, contaminated by
    wind chop on EURO) when the dashboard shows primary-swell partition
    height produces train/serve skew and hallucinated big-swell
    predictions. This function is the single integration point that
    guarantees dashboard-identity for every GFS/EURO feature CSC sees.

    Resulting per-row per-model columns (e.g. `gfs_*`):
      gfs_wave_height     — PRIMARY-swell Hs in meters (dashboard's
                             wave_height_ft / 3.28084)
      gfs_wave_period     — PRIMARY-swell period in seconds
      gfs_wave_direction  — PRIMARY-swell direction in degrees
      gfs_swell_wave_*    — same as above (redundant by design;
                             kept for features.py backwards-compat)
      gfs_secondary_swell_wave_*  — dashboard's components[1], if any
      gfs_combined_wave_height    — original combined Hs (m), kept as a
                             DIAGNOSTIC, NOT used as a feature by default
    Rows become NaN for all model-derived columns when the dashboard
    would display nothing (e.g. EURO wind-chop hours with peak period
    < 6 s). LightGBM handles NaN natively; Ridge uses its imputer.
    """
    if forecasts.empty:
        return forecasts
    from csc.dashboardify import dashboardify

    df = forecasts.copy()
    df["colname"] = df["model"].str.lower() + "_" + df["variable"]
    wide = df.pivot_table(
        index=["buoy_id", "valid_utc", "lead_days"],
        columns="colname",
        values="value",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None

    # For each row, reconstruct the raw Open-Meteo-shaped dict per model,
    # pass through dashboardify, and overwrite the feature columns with
    # the dashboard-transformed values.
    for prefix in ("gfs", "euro"):
        # Gather source columns present in this frame (graceful degradation
        # for archives that predate the 13-var schema).
        raw_cols = {v: f"{prefix}_{v}" for v in _RAW_OM_VARS
                    if f"{prefix}_{v}" in wide.columns}
        if not raw_cols:
            continue

        # Display columns we'll populate per row. Keep combined height
        # as a diagnostic column (suffixed _combined) so we can debug
        # divergences later without ever feeding it to the model.
        display_cols = {
            "wave_height", "wave_period", "wave_direction",
            "swell_wave_height", "swell_wave_period", "swell_wave_direction",
            "secondary_swell_wave_height", "secondary_swell_wave_period",
            "secondary_swell_wave_direction",
        }
        # Pre-stash the raw combined wave_height under a diagnostic name
        # before we overwrite the feature column.
        if f"{prefix}_wave_height" in wide.columns:
            wide[f"{prefix}_combined_wave_height"] = wide[f"{prefix}_wave_height"]
            wide[f"{prefix}_combined_wave_period"] = wide.get(f"{prefix}_wave_period")
            wide[f"{prefix}_combined_wave_direction"] = wide.get(f"{prefix}_wave_direction")

        # Row-wise dashboardify. pandas DataFrame.apply(axis=1) here —
        # volume is ~1M rows, call is cheap (dashboardify wraps one dict,
        # delegates to waves._parse_response).
        def _transform(row):
            raw = {"time": row["valid_utc"].isoformat()
                   if hasattr(row["valid_utc"], "isoformat") else str(row["valid_utc"])}
            for om_var, col in raw_cols.items():
                raw[om_var] = row.get(col)
            disp = dashboardify(raw) or {}
            FT_TO_M = 1.0 / 3.28084
            out = {}
            # Primary partition → wave_height (m), wave_period (s), wave_direction (°)
            hs_ft = disp.get("wave_height_ft")
            out[f"{prefix}_wave_height"] = (hs_ft * FT_TO_M) if hs_ft is not None else None
            out[f"{prefix}_wave_period"] = disp.get("wave_period_s")
            out[f"{prefix}_wave_direction"] = disp.get("wave_direction_deg")
            # Redundant swell_wave_* kept for features.py compat; same values.
            out[f"{prefix}_swell_wave_height"] = out[f"{prefix}_wave_height"]
            out[f"{prefix}_swell_wave_period"] = out[f"{prefix}_wave_period"]
            out[f"{prefix}_swell_wave_direction"] = out[f"{prefix}_wave_direction"]
            # Secondary partition → components[1] if present
            comps = disp.get("components") or []
            if len(comps) >= 2:
                c2 = comps[1]
                out[f"{prefix}_secondary_swell_wave_height"] = (
                    c2["height_ft"] * FT_TO_M if c2.get("height_ft") is not None else None)
                out[f"{prefix}_secondary_swell_wave_period"] = c2.get("period_s")
                out[f"{prefix}_secondary_swell_wave_direction"] = c2.get("direction_deg")
            else:
                out[f"{prefix}_secondary_swell_wave_height"] = None
                out[f"{prefix}_secondary_swell_wave_period"] = None
                out[f"{prefix}_secondary_swell_wave_direction"] = None
            return pd.Series(out)

        transformed = wide.apply(_transform, axis=1)
        for col in transformed.columns:
            wide[col] = transformed[col]

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
