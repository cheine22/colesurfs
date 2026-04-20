"""CSC bakeoff — 4-fold month-based cross-validation across every
candidate, plus a final refit on the full dataset for shipping artifacts.

Replaces the old single time-holdout flow. Each fold's test set contains
at least one month from every meteorological season, so reported metrics
are not an artifact of one seasonal window. Per-coast specialists are
evaluated only on their matching coast; their `scope` key marks that.

Run:
    python -m csc.experiment [--out .csc_models/2026-04-19_exp1]
"""

from __future__ import annotations

import argparse
import html
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

from csc.cv import fold_month_report, kfold_month_split
from csc.data import build_training_frame, build_training_frame_primary
from csc.evaluate import HS_THRESHOLDS_FT, summarize
from csc.features import add_engineered, feature_columns
from csc.funplus import LGBMFunPlus
from csc.models import (
    LGBMCSC, MeanBaseline, PersistenceBaseline, RawEUROBaseline,
    RawEUROPrimaryBaseline, RawGFSBaseline, RawGFSPrimaryBaseline, RidgeMOS,
    TARGET_COS, TARGET_HS, TARGET_SIN, TARGET_TP, reconstruct_dp,
)
from csc.per_coast import (
    EAST_BUOYS, WEST_BUOYS, LGBMEastCoast, LGBMWestCoast, PerCoastRouter,
    coast_mask, coast_of,
)
from csc.schema import CSC_MODELS_DIR


# ─── Legacy helper kept for funplus.evaluate_artifact back-compat ─────────

def time_holdout_split(df: pd.DataFrame, test_frac: float = 0.20
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Deprecated — the CV flow supersedes this for production training.
    Left in place because csc.funplus.evaluate_artifact still calls it as
    a cheap way to reconstruct a dataset split from a saved artifact."""
    ts = df["valid_utc"].sort_values().reset_index(drop=True)
    cutoff = ts.iloc[int(len(ts) * (1 - test_frac))]
    train = df[df["valid_utc"] < cutoff].reset_index(drop=True)
    test = df[df["valid_utc"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


# ─── Candidate registry ───────────────────────────────────────────────────


def _candidates() -> dict[str, Any]:
    return {
        "raw_gfs":        RawGFSBaseline(),
        "raw_euro":       RawEUROBaseline(),
        "mean":           MeanBaseline(),
        "persistence":    PersistenceBaseline(),
        "ridge_mos":      RidgeMOS(alpha=1.0),
        "lgbm":           LGBMCSC(),
        "funplus":        LGBMFunPlus(fun_plus_weight=3.0),
        "lgbm_east":      LGBMEastCoast(),
        "lgbm_west":      LGBMWestCoast(),
        "lgbm_per_coast": PerCoastRouter(),
    }


# Which subset of the test set each model should be evaluated on.
# "all" — every row; "east"/"west" — only matching-coast rows.
MODEL_SCOPES: dict[str, str] = {
    "raw_gfs": "all", "raw_euro": "all", "mean": "all",
    "persistence": "all", "ridge_mos": "all", "lgbm": "all",
    "funplus": "all",
    "lgbm_east": "east",
    "lgbm_west": "west",
    "lgbm_per_coast": "all",
}


def _scope_mask(X: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(X), dtype=bool)
    if scope in ("east", "west"):
        return coast_mask(X["buoy_id"], scope).to_numpy()
    raise ValueError(f"unknown scope: {scope}")


# ─── Metric aggregation across folds ──────────────────────────────────────


_NON_METRIC_KEYS = {"fold", "n", "scope"}


def _numeric_keys(d: dict[str, Any]) -> list[str]:
    out = []
    for k, v in d.items():
        if k in _NON_METRIC_KEYS:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(k)
    return out


def _aggregate_folds(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    """Given a list of per-fold metric dicts for one model, return a dict
    where each numeric metric has its mean AND a `<k>_std` sibling."""
    if not per_fold:
        return {"folds": []}
    # Collect numeric keys that appear in at least one fold
    keys: set[str] = set()
    for f in per_fold:
        keys.update(_numeric_keys(f))
    agg: dict[str, Any] = {"folds": per_fold}
    for k in keys:
        vals = []
        for f in per_fold:
            v = f.get(k)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(float(v))
        if not vals:
            agg[k] = float("nan")
            agg[f"{k}_std"] = float("nan")
            continue
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
    return agg


# ─── One-fold fit/eval ────────────────────────────────────────────────────


def _fit_one(model: Any, name: str, X_train: pd.DataFrame,
             y_train: pd.DataFrame, obs_for_persistence: pd.DataFrame) -> Any:
    if isinstance(model, PersistenceBaseline):
        model.fit(X_train, y_train, obs_df=obs_for_persistence)
    else:
        model.fit(X_train, y_train)
    return model


def _eval_on_test(model: Any, name: str, test: pd.DataFrame,
                  scope: str,
                  baseline_hs_col: str = "gfs_wave_height",
                  baseline_tp_col: str = "gfs_wave_period") -> dict[str, Any]:
    mask = _scope_mask(test, scope)
    X_test = test[mask].reset_index(drop=True)
    if len(X_test) == 0:
        return {"error": f"no test rows for scope={scope}"}
    preds = model.predict(X_test)
    eval_df = pd.concat([X_test[["obs_hs_m", "obs_tp_s", "obs_dp_deg"]]
                         .reset_index(drop=True),
                         preds.reset_index(drop=True)], axis=1)
    baseline_hs = (X_test[baseline_hs_col].to_numpy()
                   if baseline_hs_col in X_test.columns else None)
    baseline_tp = (X_test[baseline_tp_col].to_numpy()
                   if baseline_tp_col in X_test.columns else None)
    metrics = summarize(eval_df, baseline_hs=baseline_hs, baseline_tp=baseline_tp)
    metrics["n"] = int(len(X_test))
    return metrics


# ─── Main bakeoff ─────────────────────────────────────────────────────────


def run_bakeoff(out_dir: Path | None = None,
                n_folds: int = 4) -> dict[str, Any]:
    out_dir = out_dir or (CSC_MODELS_DIR /
                          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}_exp")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[exp] output dir: {out_dir}")

    df = build_training_frame()
    if df.empty:
        raise SystemExit("No training frame — run `python -m csc.backfill_primary_swell` first.")

    df = add_engineered(df)
    drop_cols = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
        "obs_hs_m", "obs_tp_s", "obs_dp_deg",
    ]
    before = len(df)
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    print(f"[exp] {before} → {len(df)} rows after dropping rows with missing core inputs")

    folds = kfold_month_split(df, n_folds=n_folds)
    for _, _, meta in folds:
        seasonal = {s: len(v) for s, v in meta["test_months_by_season"].items()}
        print(f"[exp] fold {meta['fold']}: train={meta['train_rows']:,} "
              f"test={meta['test_rows']:,}  months/season={seasonal}  "
              f"test_months={meta['test_months']}")

    per_model_folds: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _candidates()
    }

    # ── Fit every candidate on every fold ───────────────────────────────
    for fold_i, (train, test, meta) in enumerate(folds):
        y_train = train[[TARGET_HS, TARGET_TP, TARGET_SIN, TARGET_COS]].copy()
        obs_for_persistence = train[["buoy_id", "valid_utc", "obs_hs_m",
                                     "obs_tp_s", "obs_dp_deg"]].copy()
        print(f"\n[exp] ─── fold {fold_i} ───")
        candidates = _candidates()
        for name, model in candidates.items():
            scope = MODEL_SCOPES[name]
            t0 = time.time()
            try:
                _fit_one(model, name, train, y_train, obs_for_persistence)
                metrics = _eval_on_test(model, name, test, scope)
            except Exception:
                traceback.print_exc()
                metrics = {"error": traceback.format_exc()}
            elapsed = time.time() - t0
            metrics["fit_seconds"] = round(elapsed, 2)
            metrics["scope"] = scope
            metrics["fold"] = fold_i
            per_model_folds[name].append(metrics)
            mae_str = (f"{metrics['hs_mae']:.3f}m"
                       if "hs_mae" in metrics and isinstance(metrics["hs_mae"], float)
                       and not math.isnan(metrics["hs_mae"]) else "—")
            print(f"[exp] fold {fold_i} {name:<16} scope={scope:<5} "
                  f"hs_mae={mae_str}  fit {elapsed:.1f}s")

    # ── Aggregate across folds ──────────────────────────────────────────
    aggregated: dict[str, dict[str, Any]] = {}
    for name, folds_list in per_model_folds.items():
        agg = _aggregate_folds(folds_list)
        agg["scope"] = MODEL_SCOPES[name]
        aggregated[name] = agg

    # ── Refit surviving candidates on the full frame for shipping ──────
    print(f"\n[exp] ─── refit on full frame ───")
    full_y = df[[TARGET_HS, TARGET_TP, TARGET_SIN, TARGET_COS]].copy()
    obs_full = df[["buoy_id", "valid_utc", "obs_hs_m",
                   "obs_tp_s", "obs_dp_deg"]].copy()
    final = _candidates()
    for name, model in final.items():
        try:
            t0 = time.time()
            _fit_one(model, name, df, full_y, obs_full)
            model.save(out_dir / name)
            print(f"[exp] refit+save {name:<16} {time.time()-t0:.1f}s")
        except Exception:
            traceback.print_exc()

    # ── Write reports ───────────────────────────────────────────────────
    fold_meta_list = [meta for _, _, meta in folds]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cv": {
            "n_folds": n_folds,
            "total_rows": int(len(df)),
            "folds": fold_meta_list,
        },
        # Back-compat fields train.py / manifest still want to read
        "train_rows": int(len(df)),
        "test_rows": int(sum(m["test_rows"] for m in fold_meta_list)),
        "holdout_cutoff": "kfold_month_cv",
        "buoys": sorted(df["buoy_id"].astype(str).unique().tolist()),
        "metrics": aggregated,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    folds_dump = {
        "generated_at": report["generated_at"],
        "cv": report["cv"],
        "per_model": {
            name: {
                "scope": MODEL_SCOPES[name],
                "folds": per_model_folds[name],
            } for name in per_model_folds
        },
    }
    (out_dir / "folds.json").write_text(json.dumps(folds_dump, indent=2))

    (out_dir / "experiment_report.html").write_text(_render_html(report))
    print(f"[exp] wrote {out_dir / 'metrics.json'}")
    print(f"[exp] wrote {out_dir / 'folds.json'}")
    print(f"[exp] wrote {out_dir / 'experiment_report.html'}")
    return report


# ─── v2 primary-swell bakeoff ─────────────────────────────────────────────


def _candidates_primary() -> dict[str, Any]:
    """Candidate set for v2 (primary-swell target). Swaps the raw GFS and
    EURO baselines for partition-aware variants so the baseline comparison
    is apples-to-apples with NDBC primary-swell Hm0. Trained models
    (ridge/lgbm/funplus/per-coast) are unchanged — they learn to predict
    `obs_hs_m` regardless of what that column actually measures, which is
    the whole point of reusing the same schema."""
    return {
        "raw_gfs":        RawGFSPrimaryBaseline(),
        "raw_euro":       RawEUROPrimaryBaseline(),
        "mean":           MeanBaseline(),
        "persistence":    PersistenceBaseline(),
        "ridge_mos":      RidgeMOS(alpha=1.0),
        "lgbm":           LGBMCSC(),
        "funplus":        LGBMFunPlus(fun_plus_weight=3.0),
        "lgbm_east":      LGBMEastCoast(),
        "lgbm_west":      LGBMWestCoast(),
        "lgbm_per_coast": PerCoastRouter(),
    }


def run_bakeoff_primary(out_dir: Path | None = None,
                        n_folds: int = 4,
                        scope: str = "east") -> dict[str, Any]:
    """v3 bakeoff targeting NDBC primary-swell Hs, scoped to East or West.

    Differences from v2:
      1. `scope` parameter filters the training frame to the East or West
         buoy set (see csc.schema.buoys_for_scope). This is the
         non-negotiable separation the user requested — East and West are
         independent training runs with independent model artifacts.
      2. Output dir defaults to the scope's models root
         (`csc.schema.models_dir_for_scope(scope)`) so East writes to
         `.csc_models/` (shared with dashboard) and West writes to
         `.csc_models_west/` (silent).

    Emits metrics.json / folds.json / experiment_report.html in the same
    shape as v2 so notify.py + dashboard read it unchanged.
    """
    from csc.schema import buoys_for_scope, models_dir_for_scope
    scope_buoys = buoys_for_scope(scope)
    root = models_dir_for_scope(scope)
    out_dir = out_dir or (root /
                          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}_{scope}_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[exp-v3 {scope}] output dir: {out_dir}")
    print(f"[exp-v3 {scope}] training buoys: {scope_buoys}")

    df = build_training_frame_primary()
    if df.empty:
        raise SystemExit(
            "No primary-swell training frame — run "
            "`python -m csc.backfill_primary_swell` first, then retry.")

    before_scope = len(df)
    df = df[df["buoy_id"].isin(scope_buoys)].reset_index(drop=True)
    print(f"[exp-v3 {scope}] {before_scope} → {len(df)} rows after scope filter")

    df = add_engineered(df)
    # For v2 we want primary-swell features to be present — don't drop rows
    # that have a valid primary-swell observation but a NaN in combined cols.
    # Keep the same core-required gate though (gfs/euro combined must exist
    # because features.add_engineered uses them for disagreement features).
    drop_cols = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
        "obs_hs_m", "obs_tp_s", "obs_dp_deg",
    ]
    before = len(df)
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    print(f"[exp-v3 {scope}] {before} → {len(df)} rows after dropping rows with missing core inputs")

    folds = kfold_month_split(df, n_folds=n_folds)
    for _, _, meta in folds:
        seasonal = {s: len(v) for s, v in meta["test_months_by_season"].items()}
        print(f"[exp-v3 {scope}] fold {meta['fold']}: train={meta['train_rows']:,} "
              f"test={meta['test_rows']:,}  months/season={seasonal}  "
              f"test_months={meta['test_months']}")

    # Use the primary-swell GFS partition (with fallback) as the skill
    # baseline so raw_gfs skill_vs_baseline = 0 by construction.
    baseline_hs_col = ("gfs_swell_wave_height"
                       if "gfs_swell_wave_height" in df.columns
                       else "gfs_wave_height")
    baseline_tp_col = ("gfs_swell_wave_period"
                       if "gfs_swell_wave_period" in df.columns
                       else "gfs_wave_period")

    # Skip cross-coast candidates that don't apply to this scope's training
    # frame (e.g. lgbm_west on an East-only run would have zero training rows).
    # Also skip the per-coast router — it's designed for the mixed-coast case
    # where both East and West sub-models have rows to fit; on an East-only
    # or West-only scope run, its complementary sub-model fits on zero rows
    # and the artifact never saves properly.
    CROSS_COAST_ROUTERS = {"lgbm_per_coast"}
    def _candidate_compatible(name: str) -> bool:
        if name in CROSS_COAST_ROUTERS:
            return False
        m_scope = MODEL_SCOPES.get(name, "all")
        if m_scope == "all":
            return True
        return m_scope == scope  # keep only the matching-coast specialist
    candidates_for_scope = {n: m for n, m in _candidates_primary().items()
                            if _candidate_compatible(n)}
    print(f"[exp-v3 {scope}] candidates: {list(candidates_for_scope)}")

    per_model_folds: dict[str, list[dict[str, Any]]] = {
        name: [] for name in candidates_for_scope
    }

    for fold_i, (train, test, meta) in enumerate(folds):
        y_train = train[[TARGET_HS, TARGET_TP, TARGET_SIN, TARGET_COS]].copy()
        obs_for_persistence = train[["buoy_id", "valid_utc", "obs_hs_m",
                                     "obs_tp_s", "obs_dp_deg"]].copy()
        print(f"\n[exp-v3 {scope}] ─── fold {fold_i} ───")
        candidates = {n: c for n, c in candidates_for_scope.items()}
        # Rebuild candidate instances per fold to avoid carrying state.
        candidates = {n: type(m)() if hasattr(type(m), "__init__") else m
                      for n, m in candidates.items()}
        for name in candidates_for_scope:
            candidates[name] = candidates.get(name) or candidates_for_scope[name]
        for name, model in candidates.items():
            model_scope = MODEL_SCOPES[name]
            t0 = time.time()
            try:
                _fit_one(model, name, train, y_train, obs_for_persistence)
                metrics = _eval_on_test(model, name, test, model_scope,
                                        baseline_hs_col=baseline_hs_col,
                                        baseline_tp_col=baseline_tp_col)
            except Exception:
                traceback.print_exc()
                metrics = {"error": traceback.format_exc()}
            elapsed = time.time() - t0
            metrics["fit_seconds"] = round(elapsed, 2)
            metrics["scope"] = model_scope
            metrics["fold"] = fold_i
            per_model_folds[name].append(metrics)
            mae_str = (f"{metrics['hs_mae']:.3f}m"
                       if "hs_mae" in metrics and isinstance(metrics["hs_mae"], float)
                       and not math.isnan(metrics["hs_mae"]) else "—")
            print(f"[exp-v3 {scope}] fold {fold_i} {name:<16} m_scope={model_scope:<5} "
                  f"hs_mae={mae_str}  fit {elapsed:.1f}s")

    aggregated: dict[str, dict[str, Any]] = {}
    for name, folds_list in per_model_folds.items():
        agg = _aggregate_folds(folds_list)
        agg["scope"] = MODEL_SCOPES[name]
        aggregated[name] = agg

    print(f"\n[exp-v3 {scope}] ─── refit on full frame ───")
    full_y = df[[TARGET_HS, TARGET_TP, TARGET_SIN, TARGET_COS]].copy()
    obs_full = df[["buoy_id", "valid_utc", "obs_hs_m",
                   "obs_tp_s", "obs_dp_deg"]].copy()
    final = {n: m for n, m in _candidates_primary().items() if _candidate_compatible(n)}
    for name, model in final.items():
        try:
            t0 = time.time()
            _fit_one(model, name, df, full_y, obs_full)
            model.save(out_dir / name)
            print(f"[exp-v3 {scope}] refit+save {name:<16} {time.time()-t0:.1f}s")
        except Exception:
            traceback.print_exc()

    fold_meta_list = [meta for _, _, meta in folds]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": "primary_swell",
        "cv": {
            "n_folds": n_folds,
            "total_rows": int(len(df)),
            "folds": fold_meta_list,
        },
        "train_rows": int(len(df)),
        "test_rows": int(sum(m["test_rows"] for m in fold_meta_list)),
        "holdout_cutoff": "kfold_month_cv",
        "buoys": sorted(df["buoy_id"].astype(str).unique().tolist()),
        "metrics": aggregated,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    folds_dump = {
        "generated_at": report["generated_at"],
        "target": "primary_swell",
        "cv": report["cv"],
        "per_model": {
            name: {
                "scope": MODEL_SCOPES[name],
                "folds": per_model_folds[name],
            } for name in per_model_folds
        },
    }
    (out_dir / "folds.json").write_text(json.dumps(folds_dump, indent=2))

    (out_dir / "experiment_report.html").write_text(_render_html(report))
    print(f"[exp-v2] wrote {out_dir / 'metrics.json'}")
    print(f"[exp-v2] wrote {out_dir / 'folds.json'}")
    print(f"[exp-v2] wrote {out_dir / 'experiment_report.html'}")
    return report


# ─── HTML report ──────────────────────────────────────────────────────────


def _render_html(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    model_names = list(metrics.keys())

    metric_cols = [
        ("hs_bias", "Hs bias (m)", ".3f"),
        ("hs_mae",  "Hs MAE (m)",  ".3f"),
        ("hs_rmse", "Hs RMSE (m)", ".3f"),
        ("hs_si",   "Hs SI",       ".3f"),
        ("hs_hh",   "Hs HH",       ".3f"),
        ("hs_r",    "Hs r",        ".3f"),
        ("hs_slope","Hs slope",    ".3f"),
        ("hs_acc",  "Hs ACC",      ".3f"),
        ("hs_nse",  "Hs NSE",      ".3f"),
        ("hs_skill_vs_baseline", "Hs skill vs GFS", "+.3f"),
        ("tp_bias", "Tp bias (s)", ".3f"),
        ("tp_mae",  "Tp MAE (s)",  ".3f"),
        ("tp_si",   "Tp SI",       ".3f"),
        ("dp_circ_mae", "Dp circ MAE (°)", ".1f"),
        ("dp_bias", "Dp bias (°)",  ".1f"),
        ("dp_vecr", "Dp vec r",    ".3f"),
        ("pod_hs_gt_4ft", "POD Hs>4ft", ".2f"),
        ("far_hs_gt_4ft", "FAR Hs>4ft", ".2f"),
        ("csi_hs_gt_4ft", "CSI Hs>4ft", ".2f"),
        ("pod_hs_gt_6ft", "POD Hs>6ft", ".2f"),
        ("far_hs_gt_6ft", "FAR Hs>6ft", ".2f"),
        ("pod_hs_gt_8ft", "POD Hs>8ft", ".2f"),
        ("far_hs_gt_8ft", "FAR Hs>8ft", ".2f"),
        ("fit_seconds", "Fit (s)",   ".1f"),
    ]

    def fmt(v, spec):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        try:
            return format(v, spec)
        except Exception:
            return str(v)

    def cell(m: dict, key: str, spec: str) -> str:
        v = m.get(key)
        std = m.get(f"{key}_std")
        if v is None:
            return "<td>—</td>"
        base = fmt(v, spec)
        if std is None or (isinstance(std, float) and math.isnan(std)) or std == 0.0:
            return f"<td>{base}</td>"
        return (f"<td>{base} "
                f"<span class='std'>±{fmt(std, spec.replace('+', ''))}</span></td>")

    rows_html = []
    for name in model_names:
        m = metrics[name]
        if "error" in m:
            cells = "".join("<td>—</td>" for _ in metric_cols)
            rows_html.append(
                f"<tr><th>{html.escape(name)}</th><td>—</td>"
                f"{cells}<td class='err'>error</td></tr>")
            continue
        scope = html.escape(str(m.get("scope", "all")))
        cells = "".join(cell(m, k, s) for k, _, s in metric_cols)
        rows_html.append(
            f"<tr><th>{html.escape(name)}</th><td>{scope}</td>{cells}<td></td></tr>"
        )

    head_row = ("<th>scope</th>"
                + "".join(f"<th title='{html.escape(k)}'>{html.escape(label)}</th>"
                          for k, label, _ in metric_cols)
                + "<th></th>")

    cv = report.get("cv", {})
    fold_lis = []
    for meta in cv.get("folds", []):
        seasonal = meta.get("test_months_by_season", {})
        seas_str = " · ".join(f"{s}:{len(v)}" for s, v in seasonal.items())
        fold_lis.append(
            f"<li><b>fold {meta['fold']}</b> — train {meta['train_rows']:,} / "
            f"test {meta['test_rows']:,} — {html.escape(seas_str)} — "
            f"{html.escape(', '.join(meta.get('test_months', [])))}</li>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>CSC experiment report (CV)</title>
<style>
  body {{ font: 13px/1.4 ui-monospace, monospace; background: #0d0d0f; color: #e8e8f0; margin: 24px; }}
  h1 {{ font-size: 18px; margin: 0 0 8px 0; color: #7c6af7; }}
  h2 {{ font-size: 14px; margin: 16px 0 6px 0; color: #a0a0b8; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 5px 9px; text-align: right; border-bottom: 1px solid #2e2e38; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #131316; color: #a0a0b8; position: sticky; top: 0; }}
  .err {{ color: #f85149; }}
  .note {{ color: #a0a0b8; font-size: 11px; margin: 10px 0; }}
  .std {{ color: #7a7a95; font-size: 10px; }}
  ul {{ margin: 4px 0 10px 18px; padding: 0; color: #c0c0d0; }}
</style>
</head>
<body>
<h1>CSC experiment report (K-fold month CV)</h1>
<div class="note">Generated {html.escape(report['generated_at'])}
 · total {report.get('train_rows', 0):,} rows
 · {cv.get('n_folds', '?')}-fold month CV
 · buoys {", ".join(html.escape(b) for b in report['buoys'])}</div>
<h2>Fold composition</h2>
<ul>{''.join(fold_lis)}</ul>
<h2>CV-averaged metrics (± 1 SD across folds)</h2>
<table>
<thead><tr><th>model</th>{head_row}</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body></html>
"""


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.experiment")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory; defaults to timestamped dir under .csc_models/")
    ap.add_argument("--n-folds", type=int, default=4)
    args = ap.parse_args()
    run_bakeoff(args.out, n_folds=args.n_folds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
