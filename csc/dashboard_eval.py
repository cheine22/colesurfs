"""CSC dashboard evaluation — compute panel data for the /csc UI.

Produces a single .csc_data/panels.json consumed by GET /api/csc/panels.
The five panels:
  1. residual_over_time — 30-day rolling Hs MAE per model, per buoy
  2. stratified         — MAE heatmaps by (Hs bin × Tp band) and per-buoy × model
  3. peak_events        — top-15 biggest-Hs events per buoy per year, ±48h window
  4. taylor             — Taylor-diagram coords (r, sigma_ratio, centered_rmse) per model
  5. threshold_curves   — POD/FAR at HS_THRESHOLDS_FT per model

All computations pool rows across the 8 buoys when the panel asks for
"global"; per-buoy slices are computed on subsets. Results are stored in
feet (Hs) and seconds (Tp) to match the rest of the dashboard.

Run:
    python -m csc.dashboard_eval --rebuild
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csc.data import build_training_frame
from csc.display_filter import is_public_buoy
from csc.evaluate import HS_THRESHOLDS_FT, far, mae, pod
from csc.experiment import time_holdout_split
from csc.features import add_engineered
from csc.models import MeanBaseline, RawEUROBaseline, RawGFSBaseline, load_model
from csc.schema import BUOYS, CSC_DATA_DIR, CSC_MODELS_DIR, PERIOD_BANDS
from csc.serve import _strip_nans
from swell_rules import CATEGORIES, categorize


M_TO_FT = 3.28084

PANELS_PATH = CSC_DATA_DIR / "panels.json"

HS_BINS_FT = [
    ("<2ft",  0.0, 2.0),
    ("2-4ft", 2.0, 4.0),
    ("4-6ft", 4.0, 6.0),
    (">6ft",  6.0, 999.0),
]


# ─── Helpers ──────────────────────────────────────────────────────────────

def _prep_frame() -> pd.DataFrame:
    df = build_training_frame()
    if df.empty:
        raise SystemExit("empty training frame — run backfill first")
    df = add_engineered(df)
    drop_cols = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
        "obs_hs_m", "obs_tp_s", "obs_dp_deg",
    ]
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    df = df.sort_values(["buoy_id", "valid_utc"]).reset_index(drop=True)
    return df


def _public_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a frame to public (East Coast) buoys for dashboard stats."""
    mask = df["buoy_id"].astype(str).map(is_public_buoy)
    return df[mask].reset_index(drop=True)


def _load_csc_model():
    current = CSC_MODELS_DIR / "current"
    if not current.exists():
        return None, None
    manifest_path = current / "manifest.json"
    if not manifest_path.exists():
        return None, None
    manifest = json.loads(manifest_path.read_text())
    winner_dir = Path(manifest["winner_dir"])
    if not winner_dir.is_absolute():
        winner_dir = current / manifest["winner"]
    try:
        model = load_model(winner_dir)
    except Exception:
        traceback.print_exc()
        return None, None
    return model, manifest


def _add_model_predictions(df: pd.DataFrame, csc_model) -> pd.DataFrame:
    """Attach predicted Hs (in meters) from every model as pred_hs_<model>."""
    out = df.copy()
    gfs_pred = RawGFSBaseline().predict(df)
    euro_pred = RawEUROBaseline().predict(df)
    mean_pred = MeanBaseline().predict(df)
    out["pred_hs_raw_gfs"] = gfs_pred["pred_hs_m"].to_numpy()
    out["pred_hs_raw_euro"] = euro_pred["pred_hs_m"].to_numpy()
    out["pred_hs_mean"] = mean_pred["pred_hs_m"].to_numpy()
    if csc_model is not None:
        try:
            csc_pred = csc_model.predict(df)
            out["pred_hs_csc"] = csc_pred["pred_hs_m"].to_numpy()
        except Exception:
            traceback.print_exc()
            out["pred_hs_csc"] = np.nan
    else:
        out["pred_hs_csc"] = np.nan
    return out


MODEL_KEYS = ["raw_gfs", "raw_euro", "mean", "csc"]


# ─── 1. Residual-over-time (rolling 30d MAE) ──────────────────────────────

def _rolling_mae_series(ts: pd.Series, obs_ft: np.ndarray,
                       pred_ft: np.ndarray, window: str = "30D") -> pd.DataFrame:
    """Index by valid_utc, compute rolling-mean |pred-obs|. Downsample to daily."""
    err = np.abs(pred_ft - obs_ft)
    s = pd.Series(err, index=pd.DatetimeIndex(ts))
    daily = s.resample("1D").mean()
    return daily.rolling(window, min_periods=7).mean()


def _residual_over_time(df: pd.DataFrame) -> dict[str, Any]:
    """Returns {global: {ts: [...], models: {k: [ft,...]}}, per_buoy: {bid: same}}."""
    obs_ft = df["obs_hs_m"].to_numpy() * M_TO_FT

    def _series_for_subset(sub: pd.DataFrame, obs_ft_sub: np.ndarray) -> dict[str, Any]:
        ts = sub["valid_utc"]
        out_series: dict[str, list[float | None]] = {}
        date_axis = None
        for k in MODEL_KEYS:
            pred_ft = sub[f"pred_hs_{k}"].to_numpy() * M_TO_FT
            roll = _rolling_mae_series(ts, obs_ft_sub, pred_ft)
            if date_axis is None:
                date_axis = [d.strftime("%Y-%m-%d") for d in roll.index]
            out_series[k] = [None if (v is None or not np.isfinite(v)) else float(v)
                             for v in roll.to_numpy()]
        return {"ts": date_axis or [], "models": out_series}

    global_blk = _series_for_subset(df, obs_ft)

    per_buoy: dict[str, Any] = {}
    for bid, sub in df.groupby("buoy_id"):
        idx = sub.index.to_numpy()
        per_buoy[str(bid)] = _series_for_subset(sub, obs_ft[idx])

    return {"global": global_blk, "per_buoy": per_buoy, "window": "30D"}


# ─── 2. Stratified heatmaps (Hs × Tp, and buoy × model) ───────────────────

def _categorize_vec(h_ft: np.ndarray, tp_s: np.ndarray) -> np.ndarray:
    """Vectorized wrapper around swell_rules.categorize — returns CATEGORIES index."""
    out = np.zeros(h_ft.shape, dtype=np.int8)
    for i in range(h_ft.size):
        h = h_ft[i]; t = tp_s[i]
        if not (np.isfinite(h) and np.isfinite(t)):
            out[i] = 0
            continue
        cat = categorize(float(h), float(t))
        out[i] = CATEGORIES.index(cat) if cat in CATEGORIES else 0
    return out


def _strat_by_category(df_test: pd.DataFrame) -> dict[str, Any]:
    """MAE grid: model (rows) × colesurfs category (cols). Observed
    (height_ft, period_s) → category via swell_rules.categorize."""
    obs_ft = df_test["obs_hs_m"].to_numpy() * M_TO_FT
    obs_tp = df_test["obs_tp_s"].to_numpy()
    cat_idx = _categorize_vec(obs_ft, obs_tp)
    by_model: dict[str, Any] = {}
    for k in MODEL_KEYS:
        pred_ft = df_test[f"pred_hs_{k}"].to_numpy() * M_TO_FT
        err = np.abs(pred_ft - obs_ft)
        mae_row: list[float | None] = []
        cnt_row: list[int] = []
        bias_row: list[float | None] = []
        for ci in range(len(CATEGORIES)):
            m = (cat_idx == ci) & np.isfinite(err)
            n = int(m.sum())
            cnt_row.append(n)
            if n:
                mae_row.append(float(np.mean(err[m])))
                bias_row.append(float(np.mean(pred_ft[m] - obs_ft[m])))
            else:
                mae_row.append(None)
                bias_row.append(None)
        by_model[k] = {"mae_ft": mae_row, "bias_ft": bias_row, "counts": cnt_row}
    return {"categories": list(CATEGORIES), "by_model": by_model}


def _strat_heatmaps(df_test: pd.DataFrame) -> dict[str, Any]:
    """MAE on held-out test split stratified by Hs bin × period band, per model.
    Also per-buoy × model MAE heatmap."""
    obs_ft = df_test["obs_hs_m"].to_numpy() * M_TO_FT
    tp_s = df_test["gfs_wave_period"].fillna(df_test["euro_wave_period"]).to_numpy()

    # Hs bin (by observation, in ft)
    hs_bin_idx = np.full(obs_ft.shape, -1, dtype=int)
    hs_labels = []
    for i, (label, lo, hi) in enumerate(HS_BINS_FT):
        mask = (obs_ft >= lo) & (obs_ft < hi)
        hs_bin_idx[mask] = i
        hs_labels.append(label)
    # Tp bin
    tp_bin_idx = np.full(obs_ft.shape, -1, dtype=int)
    tp_labels = []
    for i, (label, lo, hi) in enumerate(PERIOD_BANDS):
        mask = (tp_s >= lo) & (tp_s < hi)
        tp_bin_idx[mask] = i
        tp_labels.append(label)

    heat_by_model: dict[str, Any] = {}
    for k in MODEL_KEYS:
        pred_ft = df_test[f"pred_hs_{k}"].to_numpy() * M_TO_FT
        err = np.abs(pred_ft - obs_ft)
        grid = [[None for _ in tp_labels] for _ in hs_labels]
        counts = [[0 for _ in tp_labels] for _ in hs_labels]
        for hi_i in range(len(hs_labels)):
            for ti_i in range(len(tp_labels)):
                m = (hs_bin_idx == hi_i) & (tp_bin_idx == ti_i) & np.isfinite(err)
                n = int(m.sum())
                counts[hi_i][ti_i] = n
                if n:
                    grid[hi_i][ti_i] = float(np.mean(err[m]))
        heat_by_model[k] = {"mae_ft": grid, "counts": counts}

    # per-buoy × model
    buoy_ids = sorted(df_test["buoy_id"].unique().tolist())
    buoy_heat: dict[str, list[float | None]] = {}
    buoy_counts: dict[str, list[int]] = {}
    for k in MODEL_KEYS:
        vals: list[float | None] = []
        cnts: list[int] = []
        pred_ft = df_test[f"pred_hs_{k}"].to_numpy() * M_TO_FT
        err = np.abs(pred_ft - obs_ft)
        for bid in buoy_ids:
            m = (df_test["buoy_id"].to_numpy() == bid) & np.isfinite(err)
            n = int(m.sum())
            cnts.append(n)
            vals.append(float(np.mean(err[m])) if n else None)
        buoy_heat[k] = vals
        buoy_counts[k] = cnts

    return {
        "hs_labels": hs_labels,
        "tp_labels": tp_labels,
        "by_model": heat_by_model,
        "by_category": _strat_by_category(df_test),
        "per_buoy": {
            "buoy_ids": buoy_ids,
            "models": MODEL_KEYS,
            "mae_ft": buoy_heat,
            "counts": buoy_counts,
        },
    }


# ─── 3. Top-N peak events per buoy per year ───────────────────────────────

def _find_peaks_min_distance(series: pd.Series, min_gap_hours: int = 48,
                             top_n: int = 15) -> list[pd.Timestamp]:
    """Greedy peak-picker: take the global max, then zero out ±min_gap around
    it, repeat until top_n peaks or no more data. Uses observed Hs series."""
    s = series.dropna().copy()
    if s.empty:
        return []
    s = s.sort_index()
    picks: list[pd.Timestamp] = []
    arr = s.copy()
    gap = pd.Timedelta(hours=min_gap_hours)
    for _ in range(top_n):
        if arr.empty:
            break
        if not np.isfinite(arr).any():
            break
        idx = arr.idxmax()
        if pd.isna(idx):
            break
        val = arr.loc[idx]
        if not np.isfinite(val):
            break
        picks.append(idx)
        mask = (arr.index >= idx - gap) & (arr.index <= idx + gap)
        arr = arr[~mask]
    return picks


SERIOUS_CATEGORIES = {"SOLID", "FIRING", "HECTIC", "MONSTRO"}


def _peak_events_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate stats over all observed peaks classified as SOLID+ on the
    colesurfs category scale. Per model report: mean/abs peak-Hs error (ft),
    mean/abs peak-timing error (h), and event-detection rate (model's peak
    within ±24h of observed peak)."""
    per_model_err: dict[str, dict[str, list[float]]] = {
        k: {"hs_err": [], "timing_err": [], "detected": []} for k in MODEL_KEYS
    }
    event_refs: list[dict[str, Any]] = []  # for an optional ref table
    n_total_peaks = 0
    for bid, sub in df.groupby("buoy_id"):
        sub = sub.sort_values("valid_utc").reset_index(drop=True)
        sub_idx = sub.set_index(pd.DatetimeIndex(sub["valid_utc"]))
        # Find peaks per year (reuse the top-N picker with large N so we
        # get every non-overlapping peak, then filter by category)
        years = sorted(sub_idx.index.year.unique().tolist())
        for yr in years:
            yr_rows = sub_idx[sub_idx.index.year == yr]
            if yr_rows.empty:
                continue
            peaks = _find_peaks_min_distance(yr_rows["obs_hs_m"],
                                             min_gap_hours=48, top_n=60)
            for peak_ts in peaks:
                lo = peak_ts - pd.Timedelta(hours=48)
                hi = peak_ts + pd.Timedelta(hours=48)
                win = sub_idx[(sub_idx.index >= lo) & (sub_idx.index <= hi)].copy()
                if win.empty:
                    continue
                obs_peak_h_ft = float(win["obs_hs_m"].max()) * M_TO_FT
                # Use observed Tp at the peak to classify
                try:
                    obs_peak_tp = float(win.loc[win["obs_hs_m"].idxmax(), "obs_tp_s"])
                except Exception:
                    obs_peak_tp = float("nan")
                if not np.isfinite(obs_peak_tp):
                    continue
                cat = categorize(obs_peak_h_ft, obs_peak_tp)
                if cat not in SERIOUS_CATEGORIES:
                    continue
                n_total_peaks += 1
                rel_hours = ((win.index - peak_ts).total_seconds() / 3600.0).to_numpy()
                obs_peak_idx = int(np.nanargmax(win["obs_hs_m"].to_numpy()))
                event_refs.append({
                    "buoy_id": str(bid),
                    "peak_ts": peak_ts.isoformat(),
                    "category": cat,
                    "obs_peak_ft": obs_peak_h_ft,
                    "obs_peak_tp_s": obs_peak_tp,
                })
                for k in MODEL_KEYS:
                    pred_ft_arr = win[f"pred_hs_{k}"].to_numpy() * M_TO_FT
                    if not np.isfinite(pred_ft_arr).any():
                        continue
                    pk_pred_val = float(np.nanmax(pred_ft_arr))
                    pk_pred_idx = int(np.nanargmax(pred_ft_arr))
                    per_model_err[k]["hs_err"].append(pk_pred_val - obs_peak_h_ft)
                    dt = float(rel_hours[pk_pred_idx] - rel_hours[obs_peak_idx])
                    per_model_err[k]["timing_err"].append(dt)
                    per_model_err[k]["detected"].append(1.0 if abs(dt) <= 24.0 else 0.0)
    summary: dict[str, Any] = {}
    for k, parts in per_model_err.items():
        hs = np.array(parts["hs_err"], dtype=float)
        tm = np.array(parts["timing_err"], dtype=float)
        dt = np.array(parts["detected"], dtype=float)
        summary[k] = {
            "n_events": int(hs.size),
            "peak_hs_bias_ft": float(np.mean(hs)) if hs.size else None,
            "peak_hs_mae_ft": float(np.mean(np.abs(hs))) if hs.size else None,
            "peak_timing_bias_h": float(np.mean(tm)) if tm.size else None,
            "peak_timing_mae_h": float(np.mean(np.abs(tm))) if tm.size else None,
            "detection_rate": float(np.mean(dt)) if dt.size else None,
        }
    event_refs.sort(key=lambda e: e["peak_ts"], reverse=True)
    return {
        "scope": "observed peak category ∈ {SOLID, FIRING, HECTIC, MONSTRO}",
        "n_events": n_total_peaks,
        "summary": summary,
        "events": event_refs[:50],   # cap reference list
    }


def _peak_events(df: pd.DataFrame, top_n: int = 15) -> dict[str, Any]:
    """For each buoy-year, find the top_n observed Hs peaks and emit ±48h
    observed + model-predicted windows."""
    result: dict[str, Any] = {}
    for bid, sub in df.groupby("buoy_id"):
        sub = sub.sort_values("valid_utc").reset_index(drop=True)
        sub_idx = sub.set_index(pd.DatetimeIndex(sub["valid_utc"]))
        years = sorted(sub_idx.index.year.unique().tolist())
        events: list[dict[str, Any]] = []
        for yr in years:
            yr_rows = sub_idx[sub_idx.index.year == yr]
            if yr_rows.empty:
                continue
            peaks = _find_peaks_min_distance(yr_rows["obs_hs_m"],
                                             min_gap_hours=48, top_n=top_n)
            for peak_ts in peaks:
                lo = peak_ts - pd.Timedelta(hours=48)
                hi = peak_ts + pd.Timedelta(hours=48)
                win = sub_idx[(sub_idx.index >= lo) & (sub_idx.index <= hi)].copy()
                if win.empty:
                    continue
                rel_hours = ((win.index - peak_ts).total_seconds() / 3600.0).tolist()
                obs_ft = (win["obs_hs_m"].to_numpy() * M_TO_FT).tolist()
                models_ft: dict[str, list[float | None]] = {}
                peak_errors: dict[str, dict[str, float | None]] = {}
                for k in MODEL_KEYS:
                    pred_ft_arr = win[f"pred_hs_{k}"].to_numpy() * M_TO_FT
                    models_ft[k] = [None if not np.isfinite(v) else float(v)
                                    for v in pred_ft_arr]
                    # peak Hs error — predicted peak vs observed peak
                    if np.isfinite(pred_ft_arr).any():
                        pk_pred_val = float(np.nanmax(pred_ft_arr))
                        pk_pred_idx = int(np.nanargmax(pred_ft_arr))
                        obs_peak_val = float(np.nanmax(win["obs_hs_m"].to_numpy())) * M_TO_FT
                        obs_peak_idx = int(np.nanargmax(win["obs_hs_m"].to_numpy()))
                        peak_errors[k] = {
                            "peak_hs_err_ft": pk_pred_val - obs_peak_val,
                            "peak_timing_err_h":
                                float(rel_hours[pk_pred_idx] - rel_hours[obs_peak_idx]),
                        }
                    else:
                        peak_errors[k] = {"peak_hs_err_ft": None,
                                          "peak_timing_err_h": None}
                events.append({
                    "peak_ts": peak_ts.isoformat(),
                    "year": int(yr),
                    "obs_peak_ft": float(win["obs_hs_m"].max()) * M_TO_FT,
                    "rel_hours": rel_hours,
                    "obs_ft": obs_ft,
                    "models_ft": models_ft,
                    "peak_errors": peak_errors,
                })
        events.sort(key=lambda e: e["obs_peak_ft"], reverse=True)
        result[str(bid)] = events
    return result


# ─── 4. Taylor diagram coords ─────────────────────────────────────────────

def _taylor_stats(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(pred)
    o = obs[m]; p = pred[m]
    if o.size < 3:
        return {"r": float("nan"), "sigma_ratio": float("nan"),
                "centered_rmse": float("nan"), "sigma_obs": float("nan"),
                "sigma_pred": float("nan")}
    sigma_o = float(np.std(o))
    sigma_p = float(np.std(p))
    r = float(np.corrcoef(o, p)[0, 1]) if sigma_o > 0 and sigma_p > 0 else float("nan")
    # centered RMSE (bias-removed)
    om = o - o.mean(); pm = p - p.mean()
    crmse = float(np.sqrt(np.mean((pm - om) ** 2)))
    return {
        "r": r,
        "sigma_obs": sigma_o,
        "sigma_pred": sigma_p,
        "sigma_ratio": sigma_p / sigma_o if sigma_o > 0 else float("nan"),
        "centered_rmse": crmse,
        "centered_rmse_ratio": crmse / sigma_o if sigma_o > 0 else float("nan"),
    }


def _taylor(df_test: pd.DataFrame) -> dict[str, Any]:
    obs = df_test["obs_hs_m"].to_numpy() * M_TO_FT
    out: dict[str, Any] = {"models": {}}
    for k in MODEL_KEYS:
        pred = df_test[f"pred_hs_{k}"].to_numpy() * M_TO_FT
        out["models"][k] = _taylor_stats(obs, pred)
    out["reference_sigma_ft"] = float(np.nanstd(obs))
    return out


# ─── 5. POD/FAR threshold curves ──────────────────────────────────────────

def _threshold_curves(df_test: pd.DataFrame) -> dict[str, Any]:
    obs_m = df_test["obs_hs_m"].to_numpy()
    thresholds_m = [t / M_TO_FT for t in HS_THRESHOLDS_FT]
    curves: dict[str, Any] = {"thresholds_ft": HS_THRESHOLDS_FT, "models": {}}
    for k in MODEL_KEYS:
        pred_m = df_test[f"pred_hs_{k}"].to_numpy()
        pod_list: list[float | None] = []
        far_list: list[float | None] = []
        for thr in thresholds_m:
            p = pod(obs_m, pred_m, thr)
            f = far(obs_m, pred_m, thr)
            pod_list.append(None if not np.isfinite(p) else float(p))
            far_list.append(None if not np.isfinite(f) else float(f))
        curves["models"][k] = {"pod": pod_list, "far": far_list}
    return curves


# ─── Orchestrator ─────────────────────────────────────────────────────────

def compute_all_panels() -> dict[str, Any]:
    t0 = time.time()
    print(f"[panels] loading frame…")
    df = _prep_frame()
    print(f"[panels] {len(df):,} rows across {df['buoy_id'].nunique()} buoys")

    csc_model, manifest = _load_csc_model()
    if csc_model is None:
        print("[panels] WARNING: no CSC artifact loadable; csc predictions will be NaN")
    else:
        print(f"[panels] loaded CSC winner {manifest.get('winner')} "
              f"(version {manifest.get('version')})")

    df = _add_model_predictions(df, csc_model)

    # Held-out split for panels 2, 4, 5
    _, test_df, cutoff = time_holdout_split(df, test_frac=0.20)
    print(f"[panels] test holdout: {len(test_df):,} rows · cutoff {cutoff}")

    # Public (East Coast) slices — the dashboard only surfaces these.
    pub_df = _public_slice(df)
    pub_test_df = _public_slice(test_df)
    print(f"[panels] public slice: {len(pub_df):,} rows ({len(pub_test_df):,} test)")

    print("[panels] 1/5 residual-over-time")
    residual = _residual_over_time(pub_df)
    print("[panels] 2/5 stratified heatmaps")
    stratified = _strat_heatmaps(pub_test_df)
    print("[panels] 3/5 peak events summary (SOLID+)")
    peaks_summary_global = _peak_events_summary(pub_df)
    peaks_summary_per_buoy: dict[str, Any] = {}
    for bid, sub in pub_df.groupby("buoy_id"):
        peaks_summary_per_buoy[str(bid)] = _peak_events_summary(sub)
    peaks = {"global": peaks_summary_global, "per_buoy": peaks_summary_per_buoy}
    print("[panels] 4/5 Taylor diagram")
    taylor = _taylor(pub_test_df)
    print("[panels] 5/5 threshold curves")
    curves = _threshold_curves(pub_test_df)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_version": (manifest or {}).get("version"),
        "artifact_winner": (manifest or {}).get("winner"),
        "holdout_cutoff": str(cutoff),
        "n_train": int(len(pub_df) - len(pub_test_df)),
        "n_test": int(len(pub_test_df)),
        "buoys": [{"buoy_id": b[0], "label": b[1]} for b in BUOYS
                  if is_public_buoy(b[0])],
        "models": MODEL_KEYS,
        "residual_over_time": residual,
        "stratified": stratified,
        "peak_events": peaks,
        "taylor": taylor,
        "threshold_curves": curves,
    }
    payload = _strip_nans(payload)
    elapsed = time.time() - t0
    print(f"[panels] done in {elapsed:.1f}s")
    return payload


def save_panels(payload: dict[str, Any], path: Path = PANELS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), default=str))
    print(f"[panels] wrote {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return path


def load_panels(path: Path = PANELS_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def panels_for_buoy(payload: dict[str, Any], buoy_id: str | None) -> dict[str, Any]:
    """Slice the full payload to a single buoy for the per-buoy dashboard.
    If buoy_id is None or unknown, return the full payload."""
    if not buoy_id:
        return payload
    out = dict(payload)
    rot = payload.get("residual_over_time") or {}
    per_buoy = (rot.get("per_buoy") or {})
    out["residual_over_time"] = {
        "global": rot.get("global"),
        "per_buoy_selected": per_buoy.get(str(buoy_id)),
        "window": rot.get("window"),
    }
    peaks = payload.get("peak_events") or {}
    per_buoy = (peaks.get("per_buoy") or {}).get(str(buoy_id))
    out["peak_events"] = {
        "buoy_id": buoy_id,
        "global": peaks.get("global"),
        "selected": per_buoy,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.dashboard_eval")
    ap.add_argument("--rebuild", action="store_true",
                    help="recompute and persist .csc_data/panels.json")
    ap.add_argument("--out", type=Path, default=PANELS_PATH)
    args = ap.parse_args()
    if not args.rebuild:
        ap.print_help()
        return 0
    payload = compute_all_panels()
    save_panels(payload, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
