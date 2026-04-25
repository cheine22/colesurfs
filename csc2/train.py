"""CSC2 trainer — fits baseline (per-cell additive bias) and ML (LightGBM)
correction models on paired EURO+GFS+buoy data for the east-coast pool.

Naming convention (per CLAUDE.md): instances are saved as
  .csc2_models/east/CSC2+{baseline|ML}_{YYMMDD}_{coverage}_v{N}/

Run from the repo root:
  python -m csc2.train --version v1
  python -m csc2.train --version v2 --no-ml          # baseline only
  python -m csc2.train --version v3 --date 260424 --coverage 0.80
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from csc2.archive_status import _obs_valid_utcs as _archive_obs_valid_utcs  # noqa: F401  # used indirectly
from csc2.schema import BUOYS, CSC2_MODELS_DIR, FORECASTS_DIR, buoys_in

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBS_HIST_DIR = PROJECT_ROOT / ".csc_data" / "observations"
OBS_LIVE_DIR = PROJECT_ROOT / ".csc_data" / "live_log" / "observations"

OUTPUT_VARS = [
    "sw1_height_ft", "sw1_period_s", "sw1_dp_sin", "sw1_dp_cos",
    "sw2_height_ft", "sw2_period_s", "sw2_dp_sin", "sw2_dp_cos",
]
SCALAR_VARS = ["sw1_height_ft", "sw1_period_s", "sw2_height_ft", "sw2_period_s"]
DIRECTIONAL_VARS = ["sw1_direction_deg", "sw2_direction_deg"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _snap_to_hour_iso(s: str) -> str | None:
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
    t = t.astimezone(timezone.utc)
    if t.minute >= 30:
        t = t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        t = t.replace(minute=0, second=0, microsecond=0)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_forecast_buoy(model: str, buoy_id: str) -> pd.DataFrame:
    root = FORECASTS_DIR / f"model={model}" / f"buoy={buoy_id}"
    if not root.exists():
        return pd.DataFrame()
    parts = []
    for shard in root.rglob("cycle=*.parquet"):
        try:
            parts.append(pd.read_parquet(shard))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["buoy_id"] = df["buoy_id"].astype(str)
    df["cycle_utc"] = df["cycle_utc"].astype(str)
    df["valid_utc"] = df["valid_utc"].astype(str)
    return df


def _read_obs_buoy(buoy_id: str) -> pd.DataFrame:
    """Return obs rows snapped-to-hour, with one row per (valid_utc_hour, partition)."""
    parts = []
    for root in (OBS_HIST_DIR, OBS_LIVE_DIR):
        p = root / f"buoy={buoy_id}"
        if not p.exists():
            continue
        for shard in p.rglob("*.parquet"):
            try:
                parts.append(pd.read_parquet(shard))
            except Exception:
                continue
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["valid_utc"])
    df["valid_utc_hour"] = df["valid_utc"].astype(str).map(_snap_to_hour_iso)
    df = df.dropna(subset=["valid_utc_hour"])
    # Take the latest ingest per (valid_utc_hour, partition) to dedupe
    df = df.sort_values("ingest_utc").drop_duplicates(
        subset=["valid_utc_hour", "partition"], keep="last"
    )
    return df


def _apply_dashboard_fallback_gfs(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror waves.py:_parse_response — when GFS swell partitions are
    absent (typical beyond ~5 day lead), synthesize sw1 from combined-sea
    fields. The dashboard does this at render time per v1.7.1; the
    trainer does it at read time so on-disk shards stay raw and the
    distinction (real partition vs. fallback) is preserved via
    combined_* and the new gfs_sw1_source tag.

    EURO (CMEMS) has no such fallback per the v1.5 honest-empty policy,
    so this only applies to GFS.
    """
    out = df.copy()
    # Coerce target columns to float64 up-front — some shards may have
    # written sw1_*_period_s/direction_deg as object dtype (mixed None +
    # numeric on rows where the GFS API returned a sparse partition).
    for col in ("gfs_sw1_height_ft", "gfs_sw1_period_s", "gfs_sw1_direction_deg",
                "gfs_combined_height_m", "gfs_combined_period_s",
                "gfs_combined_direction_deg"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    sw1 = out["gfs_sw1_height_ft"].to_numpy()
    cmb_h = out["gfs_combined_height_m"].to_numpy()
    cmb_p = out["gfs_combined_period_s"].to_numpy()
    cmb_d = out["gfs_combined_direction_deg"].to_numpy()

    import numpy as _np
    null_sw1 = _np.isnan(sw1)
    has_combined = ~_np.isnan(cmb_h) & (cmb_h > 0)
    fb = null_sw1 & has_combined

    sw1_p = out["gfs_sw1_period_s"].to_numpy().copy()
    sw1_d = out["gfs_sw1_direction_deg"].to_numpy().copy()
    sw1_h = sw1.copy()
    sw1_h[fb] = cmb_h[fb] * 3.28084
    sw1_p[fb] = cmb_p[fb]
    sw1_d[fb] = cmb_d[fb]
    out["gfs_sw1_height_ft"] = sw1_h
    out["gfs_sw1_period_s"] = sw1_p
    out["gfs_sw1_direction_deg"] = sw1_d

    src = _np.array(["partition"] * len(out), dtype=object)
    src[fb] = "combined_fallback"
    src[null_sw1 & ~has_combined] = "missing"
    out["gfs_sw1_source"] = src
    return out


def build_paired_dataset(scope: str = "east") -> pd.DataFrame:
    """Lead-aligned join of EURO + GFS + buoy obs across all scope buoys.

    One row per (buoy, cycle_utc, lead_hours) where:
      - both EURO and GFS shards have a forecast row at that valid_utc
      - the buoy has a snapped-hour partition=1 obs (primary swell target)
        OR partition=2 (secondary). Rows where neither is present are dropped.

    GFS rows where partitions are absent are reconstituted from combined-sea
    via `_apply_dashboard_fallback_gfs` to match dashboard rendering.
    """
    frames = []
    for buoy_id in buoys_in(scope):
        euro = _read_forecast_buoy("EURO", buoy_id)
        gfs = _read_forecast_buoy("GFS", buoy_id)
        if euro.empty or gfs.empty:
            continue
        obs = _read_obs_buoy(buoy_id)
        if obs.empty:
            continue

        # Lead-align EURO and GFS on the same (cycle_utc, valid_utc).
        # Include combined_* columns for the dashboard fallback (GFS only).
        keep_cols = [
            "buoy_id", "cycle_utc", "valid_utc", "lead_hours",
            "sw1_height_ft", "sw1_period_s", "sw1_direction_deg",
            "sw2_height_ft", "sw2_period_s", "sw2_direction_deg",
            "combined_height_m", "combined_period_s", "combined_direction_deg",
        ]
        e = euro[keep_cols].add_prefix("euro_").rename(
            columns={"euro_buoy_id": "buoy_id", "euro_cycle_utc": "cycle_utc",
                     "euro_valid_utc": "valid_utc", "euro_lead_hours": "lead_hours"}
        )
        g = gfs[keep_cols].add_prefix("gfs_").rename(
            columns={"gfs_buoy_id": "buoy_id", "gfs_cycle_utc": "cycle_utc",
                     "gfs_valid_utc": "valid_utc", "gfs_lead_hours": "lead_hours"}
        )
        merged = e.merge(g, on=["buoy_id", "cycle_utc", "valid_utc", "lead_hours"], how="inner")
        if merged.empty:
            continue

        # Apply GFS dashboard fallback before downstream feature engineering.
        merged = _apply_dashboard_fallback_gfs(merged)

        # Snap forecast valid_utc to the hour for joining with obs (forecast
        # valid_utcs are already on the hour but normalize anyway).
        merged["valid_utc_hour"] = merged["valid_utc"].map(_snap_to_hour_iso)

        # Pivot obs partitions 1 & 2 wide.
        for part, prefix in ((1, "obs_sw1"), (2, "obs_sw2")):
            sub = obs[obs["partition"] == part][
                ["valid_utc_hour", "hs_m", "tp_s", "dp_deg"]
            ].copy()
            sub = sub.rename(columns={
                "hs_m": f"{prefix}_height_m",
                "tp_s": f"{prefix}_period_s",
                "dp_deg": f"{prefix}_direction_deg",
            })
            merged = merged.merge(sub, on="valid_utc_hour", how="left")
            merged[f"{prefix}_height_ft"] = merged[f"{prefix}_height_m"] * 3.28084

        # Drop rows with no usable target — both sw1 and sw2 obs missing.
        has_sw1 = merged[["obs_sw1_height_ft", "obs_sw1_period_s", "obs_sw1_direction_deg"]].notna().all(axis=1)
        has_sw2 = merged[["obs_sw2_height_ft", "obs_sw2_period_s", "obs_sw2_direction_deg"]].notna().all(axis=1)
        merged = merged[has_sw1 | has_sw2].copy()

        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers — circular math, feature engineering
# ---------------------------------------------------------------------------

def _deg_to_sincos(deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    rad = np.deg2rad(deg.astype(float))
    return np.sin(rad), np.cos(rad)


def _sincos_to_deg(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0


def _circular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed angular difference a - b in degrees, in [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _doy_features(valid_utc: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    t = pd.to_datetime(valid_utc, utc=True, errors="coerce")
    doy = t.dt.dayofyear.fillna(0).to_numpy()
    th = 2 * math.pi * doy / 365.0
    return np.sin(th), np.cos(th)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Sin/cos directions for both models, both swells.
    for src in ("euro", "gfs"):
        for sw in ("sw1", "sw2"):
            s, c = _deg_to_sincos(out[f"{src}_{sw}_direction_deg"])
            out[f"{src}_{sw}_dp_sin"] = s
            out[f"{src}_{sw}_dp_cos"] = c
    # Deltas EURO − GFS for scalar swell variables.
    for sw in ("sw1", "sw2"):
        out[f"d_{sw}_height_ft"] = out[f"euro_{sw}_height_ft"] - out[f"gfs_{sw}_height_ft"]
        out[f"d_{sw}_period_s"]  = out[f"euro_{sw}_period_s"]  - out[f"gfs_{sw}_period_s"]
        # Circular delta for direction.
        out[f"d_{sw}_dp_deg"] = _circular_diff_deg(
            out[f"euro_{sw}_direction_deg"].to_numpy(dtype=float),
            out[f"gfs_{sw}_direction_deg"].to_numpy(dtype=float),
        )
    # DOY features.
    s, c = _doy_features(out["valid_utc"])
    out["doy_sin"] = s
    out["doy_cos"] = c
    # Buoy one-hot.
    east_ids = buoys_in("east")
    for bid in east_ids:
        out[f"buoy_{bid}"] = (out["buoy_id"].astype(str) == bid).astype(float)
    return out


def make_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sin/cos targets for direction; pass through scalar targets."""
    out = df.copy()
    for sw in ("sw1", "sw2"):
        s, c = _deg_to_sincos(out[f"obs_{sw}_direction_deg"])
        out[f"obs_{sw}_dp_sin"] = s
        out[f"obs_{sw}_dp_cos"] = c
    return out


# ---------------------------------------------------------------------------
# Baseline: per-(buoy × lead_hour × variable) additive bias
# ---------------------------------------------------------------------------

def fit_baseline(train: pd.DataFrame) -> dict:
    """Bias table indexed by (buoy_id, lead_hours, variable).

    Predictions for scalar vars use the EURO/GFS arithmetic mean as the
    pre-correction floor; predictions for dp use the circular mean of EURO
    and GFS dp via sin/cos averaging. The stored bias is mean(obs - pred)
    for scalars, and the (sin_bias, cos_bias) pair for dp.
    """
    bias: dict = {"scalar": {}, "dp": {}}
    # Scalar vars
    for var in ("sw1_height_ft", "sw1_period_s", "sw2_height_ft", "sw2_period_s"):
        floor = (train[f"euro_{var}"] + train[f"gfs_{var}"]) / 2.0
        target_col = f"obs_{var}"
        mask = train[target_col].notna() & floor.notna()
        if not mask.any():
            continue
        d = (train.loc[mask, target_col] - floor[mask])
        # Group by (buoy, lead) and compute mean residual.
        gb = pd.DataFrame({
            "buoy_id": train.loc[mask, "buoy_id"].astype(str),
            "lead_hours": train.loc[mask, "lead_hours"].astype(int),
            "resid": d,
        }).groupby(["buoy_id", "lead_hours"])["resid"].mean()
        bias["scalar"][var] = gb.to_dict()

    # Direction (dp): bias on sin/cos of (obs - mean_pred).
    for sw in ("sw1", "sw2"):
        # Mean predicted dp: circular mean of euro and gfs (via sin/cos).
        s_pred = (np.sin(np.deg2rad(train[f"euro_{sw}_direction_deg"]))
                  + np.sin(np.deg2rad(train[f"gfs_{sw}_direction_deg"]))) / 2.0
        c_pred = (np.cos(np.deg2rad(train[f"euro_{sw}_direction_deg"]))
                  + np.cos(np.deg2rad(train[f"gfs_{sw}_direction_deg"]))) / 2.0
        pred_dp = (np.rad2deg(np.arctan2(s_pred, c_pred)) + 360.0) % 360.0
        diff = _circular_diff_deg(
            train[f"obs_{sw}_direction_deg"].to_numpy(dtype=float),
            pred_dp.to_numpy(dtype=float) if hasattr(pred_dp, "to_numpy") else np.asarray(pred_dp),
        )
        mask = ~np.isnan(diff)
        if not mask.any():
            continue
        s_d = np.sin(np.deg2rad(diff[mask]))
        c_d = np.cos(np.deg2rad(diff[mask]))
        gb_df = pd.DataFrame({
            "buoy_id": train.loc[mask, "buoy_id"].astype(str).to_numpy(),
            "lead_hours": train.loc[mask, "lead_hours"].astype(int).to_numpy(),
            "s_d": s_d,
            "c_d": c_d,
        })
        gb = gb_df.groupby(["buoy_id", "lead_hours"]).agg({"s_d": "mean", "c_d": "mean"})
        bias["dp"][sw] = {(b, int(l)): (float(s), float(c)) for (b, l), (s, c) in gb.iterrows()}
    return bias


def predict_baseline(df: pd.DataFrame, bias: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for var in ("sw1_height_ft", "sw1_period_s", "sw2_height_ft", "sw2_period_s"):
        floor = (df[f"euro_{var}"] + df[f"gfs_{var}"]) / 2.0
        b_map = bias["scalar"].get(var, {})
        keys = list(zip(df["buoy_id"].astype(str), df["lead_hours"].astype(int)))
        b = np.array([b_map.get(k, 0.0) for k in keys], dtype=float)
        out[f"pred_{var}"] = floor.to_numpy() + b
    for sw in ("sw1", "sw2"):
        s_pred = (np.sin(np.deg2rad(df[f"euro_{sw}_direction_deg"]))
                  + np.sin(np.deg2rad(df[f"gfs_{sw}_direction_deg"]))) / 2.0
        c_pred = (np.cos(np.deg2rad(df[f"euro_{sw}_direction_deg"]))
                  + np.cos(np.deg2rad(df[f"gfs_{sw}_direction_deg"]))) / 2.0
        b_map = bias["dp"].get(sw, {})
        keys = list(zip(df["buoy_id"].astype(str), df["lead_hours"].astype(int)))
        s_b = np.array([b_map.get(k, (0.0, 1.0))[0] for k in keys], dtype=float)
        c_b = np.array([b_map.get(k, (0.0, 1.0))[1] for k in keys], dtype=float)
        # Add bias as small rotation: (sin(p+b), cos(p+b)) ≈ rotate by atan2(s_b, c_b).
        bias_deg = (np.rad2deg(np.arctan2(s_b, c_b)) + 360.0) % 360.0
        pred_dp_floor = (np.rad2deg(np.arctan2(s_pred, c_pred)) + 360.0) % 360.0
        out[f"pred_{sw}_direction_deg"] = (pred_dp_floor + bias_deg) % 360.0
    return out


# ---------------------------------------------------------------------------
# ML: LightGBM per output variable
# ---------------------------------------------------------------------------

def fit_ml(train: pd.DataFrame) -> dict:
    import lightgbm as lgb

    east_ids = buoys_in("east")
    feat_cols = [
        "lead_hours", "doy_sin", "doy_cos",
    ] + [f"buoy_{b}" for b in east_ids]
    for src in ("euro", "gfs"):
        for sw in ("sw1", "sw2"):
            feat_cols += [
                f"{src}_{sw}_height_ft", f"{src}_{sw}_period_s",
                f"{src}_{sw}_dp_sin", f"{src}_{sw}_dp_cos",
            ]
    for sw in ("sw1", "sw2"):
        feat_cols += [f"d_{sw}_height_ft", f"d_{sw}_period_s", f"d_{sw}_dp_deg"]

    X = train[feat_cols].astype(float)

    target_specs = []
    for var in ("sw1_height_ft", "sw1_period_s", "sw2_height_ft", "sw2_period_s"):
        target_specs.append((var, f"obs_{var}"))
    for sw in ("sw1", "sw2"):
        target_specs.append((f"{sw}_dp_sin", f"obs_{sw}_dp_sin"))
        target_specs.append((f"{sw}_dp_cos", f"obs_{sw}_dp_cos"))

    boosters: dict = {"feature_cols": feat_cols, "models": {}}
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        seed=42,
        feature_fraction_seed=42,
        bagging_seed=42,
        deterministic=True,
        verbose=-1,
    )
    for name, target_col in target_specs:
        y = train[target_col]
        mask = y.notna() & X.notna().all(axis=1)
        if mask.sum() < 100:
            continue
        ds = lgb.Dataset(X[mask].to_numpy(), label=y[mask].to_numpy(), feature_name=feat_cols)
        booster = lgb.train(params, ds, num_boost_round=400)
        boosters["models"][name] = booster
    return boosters


def predict_ml(df: pd.DataFrame, boosters: dict) -> pd.DataFrame:
    feat_cols = boosters["feature_cols"]
    X = df[feat_cols].astype(float).to_numpy()
    out = pd.DataFrame(index=df.index)
    for name, model in boosters["models"].items():
        pred = model.predict(X)
        if name in ("sw1_height_ft", "sw1_period_s", "sw2_height_ft", "sw2_period_s"):
            out[f"pred_{name}"] = pred
        else:
            out[f"pred_{name}"] = pred  # sin/cos pred
    # Recombine sin/cos predictions into degrees per swell.
    for sw in ("sw1", "sw2"):
        s_col = f"pred_{sw}_dp_sin"
        c_col = f"pred_{sw}_dp_cos"
        if s_col in out and c_col in out:
            out[f"pred_{sw}_direction_deg"] = _sincos_to_deg(
                out[s_col].to_numpy(), out[c_col].to_numpy()
            )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_set(df: pd.DataFrame, pred_df: pd.DataFrame, model_name: str) -> dict:
    """Return MAE/RMSE/bias for each variable (vs obs), plus surfer metrics
    (sensitivity / specificity / PPV / NPV per swell category) on primary
    swell categorization."""
    out = {}
    for var in ("sw1_height_ft", "sw1_period_s", "sw2_height_ft", "sw2_period_s"):
        if f"pred_{var}" not in pred_df:
            continue
        y = df[f"obs_{var}"]
        p = pred_df[f"pred_{var}"]
        m = y.notna() & p.notna()
        if m.sum() == 0:
            continue
        d = (p[m] - y[m]).to_numpy()
        out[var] = {
            "mae": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "bias": float(np.mean(d)),
            "n": int(m.sum()),
        }
    for sw in ("sw1", "sw2"):
        var = f"{sw}_direction_deg"
        if f"pred_{var}" not in pred_df:
            continue
        y = df[f"obs_{var}"]
        p = pred_df[f"pred_{var}"]
        m = y.notna() & p.notna()
        if m.sum() == 0:
            continue
        d = _circular_diff_deg(p[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float))
        out[var] = {
            "mae": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "bias": float(np.mean(d)),
            "n": int(m.sum()),
        }
    out["surfer"] = surfer_metric_set(df, pred_df)
    return out


SURFER_CATEGORIES = ("FUN", "SOLID", "FIRING", "HECTIC", "MONSTRO")
FUN_OR_BETTER = set(SURFER_CATEGORIES)  # everything from FUN up


def surfer_metric_set(df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Sensitivity / specificity / PPV / NPV for the dashboard's swell
    categorizer applied to primary-swell predictions vs observations.

    Categories: each individual {FUN, SOLID, FIRING, HECTIC, MONSTRO}
    plus a combined "FUN_OR_BETTER" positive class. Categorization uses
    `swell_rules.categorize(height_ft, period_s)` — the exact same
    function the dashboard uses for cell coloring, so a correct
    classification here lines up byte-for-byte with what a user would
    have seen.

    Returns: {category: {sens, spec, ppv, npv, tp, fp, tn, fn, n}}
    """
    if "pred_sw1_height_ft" not in pred_df or "pred_sw1_period_s" not in pred_df:
        return {}
    import swell_rules

    y_h = df["obs_sw1_height_ft"]
    y_p = df["obs_sw1_period_s"]
    p_h = pred_df["pred_sw1_height_ft"]
    p_p = pred_df["pred_sw1_period_s"]
    mask = y_h.notna() & y_p.notna() & p_h.notna() & p_p.notna()
    if mask.sum() == 0:
        return {}

    y_cat = [swell_rules.categorize(float(h), float(p)) for h, p in
             zip(y_h[mask].to_numpy(), y_p[mask].to_numpy())]
    p_cat = [swell_rules.categorize(float(h), float(p)) for h, p in
             zip(p_h[mask].to_numpy(), p_p[mask].to_numpy())]

    out: dict = {}
    cats = list(SURFER_CATEGORIES) + ["FUN_OR_BETTER"]
    for cat in cats:
        if cat == "FUN_OR_BETTER":
            y_pos = [c in FUN_OR_BETTER for c in y_cat]
            p_pos = [c in FUN_OR_BETTER for c in p_cat]
        else:
            y_pos = [c == cat for c in y_cat]
            p_pos = [c == cat for c in p_cat]
        tp = sum(1 for yi, pi in zip(y_pos, p_pos) if yi and pi)
        fp = sum(1 for yi, pi in zip(y_pos, p_pos) if (not yi) and pi)
        fn = sum(1 for yi, pi in zip(y_pos, p_pos) if yi and (not pi))
        tn = sum(1 for yi, pi in zip(y_pos, p_pos) if (not yi) and (not pi))
        n_pos = tp + fn
        n_neg = tn + fp
        sens = (tp / n_pos) if n_pos else None
        spec = (tn / n_neg) if n_neg else None
        ppv  = (tp / (tp + fp)) if (tp + fp) else None
        npv  = (tn / (tn + fn)) if (tn + fn) else None
        out[cat] = {
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(mask.sum()),
        }
    return out


def raw_predictions(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """For comparison: a model where 'pred' = raw EURO or raw GFS."""
    out = pd.DataFrame(index=df.index)
    for var in ("sw1_height_ft", "sw1_period_s", "sw2_height_ft", "sw2_period_s"):
        out[f"pred_{var}"] = df[f"{source}_{var}"]
    for sw in ("sw1", "sw2"):
        out[f"pred_{sw}_direction_deg"] = df[f"{source}_{sw}_direction_deg"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_today_yymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%y%m%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="east", choices=["east", "west"])
    ap.add_argument("--date", default=None, help="YYMMDD train date (default: today UTC)")
    ap.add_argument("--coverage", default=None, type=float,
                    help="Coverage fraction to embed in name (default: pull from archive_status_cache)")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--holdout-frac", default=0.15, type=float)
    args = ap.parse_args()

    date_str = args.date or _format_today_yymmdd()

    if args.coverage is None:
        from csc2.archive_status import summarize
        payload = summarize()
        cov_doys = len(payload["histograms"]["combined_east"]["paired_by_doy"])
        cov_frac = round(cov_doys / 365.0, 2)
    else:
        cov_frac = round(args.coverage, 2)

    cov_str = f"{cov_frac:.2f}"

    print(f"[load] building paired east-pool dataset…")
    raw = build_paired_dataset(args.scope)
    if raw.empty:
        raise SystemExit("no paired data found")
    print(f"[load] {len(raw):,} paired rows across {raw.buoy_id.nunique()} buoys, "
          f"cycles {raw.cycle_utc.min()}..{raw.cycle_utc.max()}")
    if "gfs_sw1_source" in raw.columns:
        src_counts = raw["gfs_sw1_source"].value_counts().to_dict()
        print(f"[load] gfs_sw1_source breakdown: {src_counts}")

    df = make_target_columns(add_features(raw))

    # Time-aware holdout: last `holdout_frac` of unique cycles by date.
    cycles = sorted(df["cycle_utc"].unique())
    n_hold = max(1, int(round(len(cycles) * args.holdout_frac)))
    hold_cycles = set(cycles[-n_hold:])
    test = df[df["cycle_utc"].isin(hold_cycles)].copy()
    train = df[~df["cycle_utc"].isin(hold_cycles)].copy()
    print(f"[split] train: {len(train):,} rows ({len(cycles)-n_hold} cycles); "
          f"test: {len(test):,} rows ({n_hold} cycles)")

    metrics = {}
    metrics["raw_EURO"] = metric_set(test, raw_predictions(test, "euro"), "raw_EURO")
    metrics["raw_GFS"] = metric_set(test, raw_predictions(test, "gfs"), "raw_GFS")

    out_root = CSC2_MODELS_DIR / args.scope
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.no_baseline:
        print("[fit] baseline (per-cell additive bias)…")
        bias = fit_baseline(train)
        pred = predict_baseline(test, bias)
        metrics["baseline"] = metric_set(test, pred, "baseline")
        name = f"CSC2+baseline_{date_str}_{cov_str}_{args.version}"
        outdir = out_root / name
        outdir.mkdir(parents=True, exist_ok=True)
        # Serialize bias table as JSON (small).
        ser = {
            "scalar": {
                v: {f"{b}|{l}": float(x) for (b, l), x in tbl.items()}
                for v, tbl in bias["scalar"].items()
            },
            "dp": {
                sw: {f"{b}|{l}": [float(s), float(c)] for (b, l), (s, c) in tbl.items()}
                for sw, tbl in bias["dp"].items()
            },
        }
        (outdir / "bias.json").write_text(json.dumps(ser, indent=2))
        meta = {
            "name": name,
            "trained_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scope": args.scope,
            "buoys": buoys_in(args.scope),
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_train_cycles": len(cycles) - n_hold,
            "n_test_cycles": n_hold,
            "coverage_doys": int(round(cov_frac * 365)),
            "coverage_frac": cov_frac,
            "gfs_sw1_source_breakdown": (
                raw["gfs_sw1_source"].value_counts().to_dict()
                if "gfs_sw1_source" in raw.columns else {}
            ),
            "metrics": {k: v for k, v in metrics.items() if k in ("raw_EURO", "raw_GFS", "baseline")},
        }
        (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[save] {outdir}")

    if not args.no_ml:
        print("[fit] ML (LightGBM, 8 boosters)…")
        boosters = fit_ml(train)
        pred = predict_ml(test, boosters)
        metrics["ml"] = metric_set(test, pred, "ml")
        name = f"CSC2+ML_{date_str}_{cov_str}_{args.version}"
        outdir = out_root / name
        outdir.mkdir(parents=True, exist_ok=True)
        for tname, model in boosters["models"].items():
            model.save_model(str(outdir / f"booster_{tname}.txt"))
        (outdir / "feature_cols.json").write_text(json.dumps(boosters["feature_cols"], indent=2))
        meta = {
            "name": name,
            "trained_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scope": args.scope,
            "buoys": buoys_in(args.scope),
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_train_cycles": len(cycles) - n_hold,
            "n_test_cycles": n_hold,
            "coverage_doys": int(round(cov_frac * 365)),
            "coverage_frac": cov_frac,
            "gfs_sw1_source_breakdown": (
                raw["gfs_sw1_source"].value_counts().to_dict()
                if "gfs_sw1_source" in raw.columns else {}
            ),
            "params": {
                "learning_rate": 0.05, "num_leaves": 63, "num_boost_round": 400,
                "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
            },
            "metrics": {k: v for k, v in metrics.items() if k in ("raw_EURO", "raw_GFS", "ml")},
        }
        (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[save] {outdir}")

    # Combined comparison table for stdout.
    print("\n=== Holdout metrics (lower MAE / RMSE = better) ===")
    cols = ["raw_EURO", "raw_GFS", "baseline", "ml"]
    cols = [c for c in cols if c in metrics]
    vars_show = ["sw1_height_ft", "sw1_period_s", "sw1_direction_deg",
                 "sw2_height_ft", "sw2_period_s", "sw2_direction_deg"]
    for v in vars_show:
        print(f"\n  {v}:")
        for stat in ("mae", "rmse", "bias", "n"):
            row = [f"{stat:>5}"]
            for c in cols:
                cell = metrics[c].get(v, {}).get(stat)
                if cell is None:
                    row.append(f"{c:>10}=  --   ")
                elif stat == "n":
                    row.append(f"{c:>10}={cell:>6}")
                else:
                    row.append(f"{c:>10}={cell:>7.3f}")
            print("    " + "  ".join(row))


if __name__ == "__main__":
    main()
