"""Feature engineering for CSC v1 (analysis-level correction).

Consumes the wide frame produced by csc.data.build_training_frame and
produces a DataFrame of features plus target columns ready for the
bakeoff and training.

For the analysis-only v1, lead_days is a constant input (no lead-time
dimension to learn), so we exclude it from features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Target columns emitted by csc.data.build_training_frame
TARGETS = ["obs_hs_m", "obs_tp_s", "obs_dp_deg"]


def _safe_div(a, b):
    return np.where(np.abs(b) > 1e-9, a / b, 0.0)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add CSC v1 features in place on a copy and return it.

    Input columns (from _wide_forecasts with backfill variables):
      gfs_wave_height, gfs_wave_period, gfs_wave_direction,
      gfs_swell_wave_height, gfs_swell_wave_period, gfs_swell_wave_direction,
      gfs_secondary_swell_wave_*,
      euro_wave_height, euro_wave_period, euro_wave_direction,
      (EURO has no swell_* partitions in Open-Meteo — those cols will be absent)
    """
    out = df.copy()
    # Ensure all expected columns exist (fill missing with NaN so downstream
    # code doesn't KeyError).
    expected_cols = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "gfs_swell_wave_height", "gfs_swell_wave_period", "gfs_swell_wave_direction",
        "gfs_secondary_swell_wave_height", "gfs_secondary_swell_wave_period",
        "gfs_secondary_swell_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
    ]
    for c in expected_cols:
        if c not in out.columns:
            out[c] = np.nan

    # Disagreement features (the likely signal)
    out["d_hs"] = out["gfs_wave_height"] - out["euro_wave_height"]
    out["abs_d_hs"] = out["d_hs"].abs()
    out["d_tp"] = out["gfs_wave_period"] - out["euro_wave_period"]
    out["abs_d_tp"] = out["d_tp"].abs()

    # Angular disagreement between model directions (shortest-arc, degrees)
    dd = (out["gfs_wave_direction"] - out["euro_wave_direction"]) % 360
    dd = np.minimum(dd, 360 - dd)
    out["abs_d_dp"] = dd

    # Mean / proxy ensemble values
    out["mean_hs"] = out[["gfs_wave_height", "euro_wave_height"]].mean(axis=1)
    out["mean_tp"] = out[["gfs_wave_period", "euro_wave_period"]].mean(axis=1)

    # Interaction features — height × period × direction regime
    out["gfs_hs_x_tp"] = out["gfs_wave_height"] * out["gfs_wave_period"]
    out["euro_hs_x_tp"] = out["euro_wave_height"] * out["euro_wave_period"]
    out["gfs_tp_sin_dp"] = out["gfs_wave_period"] * np.sin(np.deg2rad(out["gfs_wave_direction"]))
    out["gfs_tp_cos_dp"] = out["gfs_wave_period"] * np.cos(np.deg2rad(out["gfs_wave_direction"]))
    out["euro_tp_sin_dp"] = out["euro_wave_period"] * np.sin(np.deg2rad(out["euro_wave_direction"]))
    out["euro_tp_cos_dp"] = out["euro_wave_period"] * np.cos(np.deg2rad(out["euro_wave_direction"]))

    # Direction as sin/cos so linear models handle wrap-around
    out["gfs_sin_dp"] = np.sin(np.deg2rad(out["gfs_wave_direction"]))
    out["gfs_cos_dp"] = np.cos(np.deg2rad(out["gfs_wave_direction"]))
    out["euro_sin_dp"] = np.sin(np.deg2rad(out["euro_wave_direction"]))
    out["euro_cos_dp"] = np.cos(np.deg2rad(out["euro_wave_direction"]))

    # Observed direction encoded for learning (target sin/cos pair)
    out["obs_sin_dp"] = np.sin(np.deg2rad(out["obs_dp_deg"]))
    out["obs_cos_dp"] = np.cos(np.deg2rad(out["obs_dp_deg"]))

    # Seasonality
    doy = out["valid_utc"].dt.dayofyear.astype(float)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["month"] = out["valid_utc"].dt.month.astype(int)

    # Period band categorical (informative for trees)
    tp = out["gfs_wave_period"].fillna(out["euro_wave_period"])
    out["period_band"] = pd.cut(
        tp,
        bins=[-0.001, 8.0, 11.0, 14.0, 1000.0],
        labels=["<8s", "8-11s", "11-14s", ">14s"],
    ).astype("object").fillna("unknown")

    return out


def feature_columns(use_partition_features: bool = True) -> list[str]:
    """Ordered list of feature column names the model sees.

    use_partition_features=True includes the GFS swell partition columns
    (Euro has no partitions in Open-Meteo). These will be NaN in some
    rows — tree models handle that natively; for Ridge we impute.
    """
    base = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
        "d_hs", "abs_d_hs", "d_tp", "abs_d_tp", "abs_d_dp",
        "mean_hs", "mean_tp",
        "gfs_hs_x_tp", "euro_hs_x_tp",
        "gfs_tp_sin_dp", "gfs_tp_cos_dp",
        "euro_tp_sin_dp", "euro_tp_cos_dp",
        "gfs_sin_dp", "gfs_cos_dp",
        "euro_sin_dp", "euro_cos_dp",
        "sin_doy", "cos_doy",
    ]
    if use_partition_features:
        base += [
            "gfs_swell_wave_height", "gfs_swell_wave_period",
            "gfs_swell_wave_direction",
            "gfs_secondary_swell_wave_height", "gfs_secondary_swell_wave_period",
            "gfs_secondary_swell_wave_direction",
        ]
    return base


def categorical_columns() -> list[str]:
    return ["buoy_id", "period_band", "month"]
