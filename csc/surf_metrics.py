"""Surfer-relevant evaluation — a compact leaderboard that scores every
candidate on what surfers and forecasters actually care about, not just
raw Hs MAE.

Why a separate module: `csc.evaluate.summarize` emits the standard
oceanography-verification bundle (bias / MAE / RMSE / SI / HH / POD / FAR /
CSI / ACC / NSE / etc.). That is the right view for model-skill journals,
but it over-weights smooth accuracy and under-weights the discrete events a
surfer cares about: "did the model miss a surfable day?", "did it cry big-
surf when it was flat?", "is the period number trustworthy?". This module
aggregates a second, surf-centric view that sits beside metrics.json.

The metrics (rewritten 2026-04 to expose full 2×2 confusion matrices):

  1. FUN-category confusion (daytime-only, per-day)
       TP: both predicted & observed peak-hour were FUN+
       FP: predicted FUN+ but observed below FUN
       FN: observed FUN+ but predicted below FUN
       TN: both below FUN
       Rates: precision=TP/(TP+FP), recall=TP/(TP+FN),
              FP rate=FP/(FP+TN), FN rate=FN/(FN+TP)

  2. SOLID+ confusion — identical structure, threshold at SOLID.

  3. Tp MAE across ALL held-out rows (no category / daytime filter).

  4. Hs MAE (ft) restricted to rows where the *observed* category is FUN+
     (24-hour, no daytime filter).

  5. general_swell_score — blend of Hs / Tp / Dp normalized against the
     `mean` baseline candidate. 1.0 = tied with baseline, 0 = perfect,
     >1 = worse than baseline.

Day-level aggregation: for the confusion matrices only, each (buoy,
calendar-day) is reduced to its PEAK daytime hour — category is taken on
(max-Hs, matched-Tp) inside the local sunrise→sunset window — and the
confusion is counted in days.

Scope: all public functions accept `scope` (default "east"). West runs
the same math but writes to a scope-suffixed output file so it never
surfaces on the dashboard.

Run:
    python -m csc.surf_metrics --scope east
    python -m csc.surf_metrics --scope west   # silent, separate outputs

Outputs, written next to the artifact dir:
    surf_metrics.json         — per-model / per-buoy numbers
                                (scope != east → surf_metrics_<scope>.json)
    ../../csc/docs/model-comparison.md     — leaderboard table (east)
    ../../csc/docs/model-comparison_<scope>.md (non-east)
    ../../csc/docs/per-buoy-percentages.md — per-buoy details (east)
    ../../csc/docs/per-buoy-percentages_<scope>.md (non-east)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astral import LocationInfo
from astral.sun import sun as astral_sun
from zoneinfo import ZoneInfo

from csc.cv import kfold_month_split
from csc.data import build_training_frame
from csc.evaluate import circular_mae, mae as eval_mae
from csc.experiment import _candidates, _scope_mask, MODEL_SCOPES, _fit_one
from csc.features import add_engineered
from csc.funplus import category_index_array
from csc.models import TARGET_COS, TARGET_HS, TARGET_SIN, TARGET_TP
from csc.schema import BUOYS, CSC_MODELS_DIR, buoys_for_scope
from swell_rules import CATEGORIES


FT_PER_M = 1.0 / 0.3048
FUN_IDX = CATEGORIES.index("FUN")
SOLID_IDX = CATEGORIES.index("SOLID")


# ─── Metric weights for the composite surfer score ────────────────────────
# Explicit ranking per the user priority order. Values are applied to each
# metric's min-max-normalized rank among models, not the raw values, so
# large absolute scales (day counts vs MAE) don't swamp each other.
COMPOSITE_WEIGHTS = {
    "fun_fn_days":            5.0,   # missed FUN+ days (observed FUN+, predicted below)
    "fun_fp_days":            3.0,   # false-alarm FUN+ days
    "solid_fn_days":          3.0,   # missed SOLID+ days
    "solid_fp_days":          2.0,   # false-alarm SOLID+ days
    "tp_mae_all":             2.0,
    "fun_plus_hs_mae_ft":     1.5,
    "general_swell_score":    1.0,
}


# ─── Buoy metadata + daytime window pre-compute ──────────────────────────


_BUOY_META = {str(bid): {"lat": lat, "lon": lon, "label": label,
                          "operator": op}
               for (bid, label, lat, lon, op) in BUOYS}


def _buoy_tz(lat: float, lon: float) -> str:
    """Rough tz pick — East-Coast buoys use America/New_York, West-Coast
    buoys use America/Los_Angeles. The daytime window only needs to be
    correct to the nearest ~30 min; either IANA zone handles DST for us."""
    return "America/New_York" if lon > -100.0 else "America/Los_Angeles"


def _sunrise_sunset_utc(lat: float, lon: float, d: date, tz_name: str
                        ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (sunrise_utc, sunset_utc) as tz-aware pandas.Timestamp at UTC.
    Returns None if astral can't compute it (polar edge cases)."""
    try:
        loc = LocationInfo(latitude=float(lat), longitude=float(lon),
                           timezone=tz_name)
        s = astral_sun(loc.observer, date=d, tzinfo=ZoneInfo(tz_name))
        sr = pd.Timestamp(s["sunrise"]).tz_convert("UTC")
        ss = pd.Timestamp(s["sunset"]).tz_convert("UTC")
        return sr, ss
    except Exception:
        return None


def _build_daytime_lookup(df: pd.DataFrame) -> dict[str, dict[date,
                                                               tuple[pd.Timestamp,
                                                                     pd.Timestamp]]]:
    """Pre-compute {buoy_id: {local_date: (sunrise_utc, sunset_utc)}}
    over the full date range present in df, keyed by the buoy's local
    calendar day (dates in the buoy's local tz)."""
    if df.empty:
        return {}
    buoys = sorted(df["buoy_id"].astype(str).unique())
    # Broad range covering every row.
    ts = pd.to_datetime(df["valid_utc"], utc=True)
    start_utc = ts.min().to_pydatetime()
    end_utc = ts.max().to_pydatetime()

    lookup: dict[str, dict[date, tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for bid in buoys:
        meta = _BUOY_META.get(bid)
        if meta is None:
            lookup[bid] = {}
            continue
        tz_name = _buoy_tz(meta["lat"], meta["lon"])
        tz = ZoneInfo(tz_name)
        # Walk each local-day from 1 day before start_utc to 1 day after
        # end_utc — this catches any UTC row whose local day sits outside
        # the raw UTC-date span (e.g. 23:00 UTC = next day local).
        start_local = start_utc.astimezone(tz).date() - timedelta(days=1)
        end_local = end_utc.astimezone(tz).date() + timedelta(days=1)
        per_day: dict[date, tuple[pd.Timestamp, pd.Timestamp]] = {}
        d = start_local
        while d <= end_local:
            win = _sunrise_sunset_utc(meta["lat"], meta["lon"], d, tz_name)
            if win is not None:
                per_day[d] = win
            d += timedelta(days=1)
        lookup[bid] = per_day
    return lookup


def _daytime_mask(buoy_ids: np.ndarray, valid_utc: pd.Series,
                  lookup: dict[str, dict[date, tuple[pd.Timestamp,
                                                      pd.Timestamp]]]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, local_day_array) for the rows.

    mask[i] is True iff valid_utc[i] falls between sunrise and sunset
    at buoy[i] on the matching local day. local_day_array[i] is the
    buoy-local calendar date for that row (as a Python date).
    """
    ts = pd.to_datetime(valid_utc, utc=True)
    n = len(ts)
    mask = np.zeros(n, dtype=bool)
    local_days = np.empty(n, dtype=object)
    # Group by buoy to share the lookup dict + tz.
    s = pd.Series(buoy_ids).astype(str)
    for bid, idx in s.groupby(s).groups.items():
        meta = _BUOY_META.get(bid)
        if meta is None:
            continue
        tz = ZoneInfo(_buoy_tz(meta["lat"], meta["lon"]))
        per_day = lookup.get(bid, {})
        sub = ts.iloc[idx]
        sub_local_dates = sub.dt.tz_convert(tz).dt.date
        for i_row, t_utc, d_local in zip(idx, sub, sub_local_dates):
            local_days[i_row] = d_local
            win = per_day.get(d_local)
            if win is None:
                continue
            sr, ss = win
            if sr <= t_utc <= ss:
                mask[i_row] = True
    return mask, local_days


# ─── Core per-buoy metric computation (scope-filtered) ────────────────────


def _daily_peaks(hs_ft: np.ndarray, tp_s: np.ndarray,
                 days: np.ndarray) -> pd.DataFrame:
    """Reduce hourly rows to per-local-day peak Hs (and its matching Tp)."""
    frame = pd.DataFrame({"day": days, "hs_ft": hs_ft, "tp_s": tp_s})
    frame = frame.dropna(subset=["day"])
    if frame.empty:
        return pd.DataFrame(columns=["day", "hs_ft", "tp_s"])
    idx = frame.groupby("day")["hs_ft"].idxmax()
    return frame.loc[idx].reset_index(drop=True)


def _confusion(obs_pos: np.ndarray, pred_pos: np.ndarray) -> dict[str, int]:
    """Return TP / FP / FN / TN counts for two boolean arrays."""
    return {
        "tp": int(( obs_pos &  pred_pos).sum()),
        "fp": int((~obs_pos &  pred_pos).sum()),
        "fn": int(( obs_pos & ~pred_pos).sum()),
        "tn": int((~obs_pos & ~pred_pos).sum()),
    }


def _confusion_rates(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    def _safe(n, d):
        return (n / d) if d > 0 else float("nan")
    return {
        "precision": _safe(tp, tp + fp),
        "recall":    _safe(tp, tp + fn),
        "fp_rate":   _safe(fp, fp + tn),
        "fn_rate":   _safe(fn, fn + tp),
        "accuracy":  _safe(tp + tn, tp + fp + fn + tn),
    }


def _merge_confusions(cs: list[dict[str, int]]) -> dict[str, int]:
    out = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for c in cs:
        for k in out:
            out[k] += int(c.get(k, 0))
    return out


def _per_buoy_metrics(
    test: pd.DataFrame,
    preds_hs_m: np.ndarray,
    preds_tp_s: np.ndarray,
    preds_dp_deg: np.ndarray,
    daytime_lookup: dict[str, dict[date, tuple[pd.Timestamp, pd.Timestamp]]],
    allowed_buoys: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute the new surf metrics per buoy on one fold's test slice.

    Returns {buoy_id: {...}}. Only buoys in `allowed_buoys` are kept.
    """
    out: dict[str, dict[str, Any]] = {}
    if len(test) == 0:
        return out

    obs_hs_ft = test["obs_hs_m"].to_numpy() * FT_PER_M
    obs_tp = test["obs_tp_s"].to_numpy()
    obs_dp = test["obs_dp_deg"].to_numpy()
    pred_hs_ft = preds_hs_m * FT_PER_M

    # Observation FUN+ mask at hourly resolution (for FUN+ Hs MAE — 24h)
    obs_cat_hr = category_index_array(obs_hs_ft, obs_tp)
    fun_plus_hourly = obs_cat_hr >= FUN_IDX

    buoy_arr = test["buoy_id"].astype(str).to_numpy()
    day_mask, local_days = _daytime_mask(buoy_arr, test["valid_utc"],
                                         daytime_lookup)

    for buoy in sorted(set(buoy_arr)):
        if buoy not in allowed_buoys:
            continue
        bmask = (buoy_arr == buoy)
        if not bmask.any():
            continue

        # Continuous metrics — 24-hour, no daytime filter
        b_obs_hs = obs_hs_ft[bmask]
        b_obs_tp = obs_tp[bmask]
        b_obs_dp = obs_dp[bmask]
        b_pred_hs = pred_hs_ft[bmask]
        b_pred_tp = preds_tp_s[bmask]
        b_pred_dp = preds_dp_deg[bmask]
        b_fun_plus_hr = fun_plus_hourly[bmask]

        tp_mae_all = float(eval_mae(b_obs_tp, b_pred_tp))
        hs_mae_ft = float(eval_mae(b_obs_hs, b_pred_hs))
        dp_cmae = float(circular_mae(b_obs_dp, b_pred_dp))
        if b_fun_plus_hr.any():
            fun_plus_hs_mae_ft = float(eval_mae(
                b_obs_hs[b_fun_plus_hr], b_pred_hs[b_fun_plus_hr]))
        else:
            fun_plus_hs_mae_ft = float("nan")

        # Day-level confusion — DAYTIME ONLY, peak daytime hour per day
        day_sub = day_mask[bmask]
        local_day_sub = local_days[bmask]
        if day_sub.any():
            obs_peaks = _daily_peaks(
                b_obs_hs[day_sub], b_obs_tp[day_sub],
                local_day_sub[day_sub])
            pred_peaks = _daily_peaks(
                b_pred_hs[day_sub], b_pred_tp[day_sub],
                local_day_sub[day_sub])
            peaks = obs_peaks.merge(
                pred_peaks, on="day", suffixes=("_obs", "_pred"), how="inner")
        else:
            peaks = pd.DataFrame(columns=[
                "day", "hs_ft_obs", "tp_s_obs", "hs_ft_pred", "tp_s_pred"])

        if len(peaks):
            obs_cat_day = category_index_array(
                peaks["hs_ft_obs"].to_numpy(), peaks["tp_s_obs"].to_numpy())
            pred_cat_day = category_index_array(
                peaks["hs_ft_pred"].to_numpy(), peaks["tp_s_pred"].to_numpy())
        else:
            obs_cat_day = np.array([], dtype=int)
            pred_cat_day = np.array([], dtype=int)

        fun_conf = _confusion(obs_cat_day >= FUN_IDX,
                              pred_cat_day >= FUN_IDX)
        solid_conf = _confusion(obs_cat_day >= SOLID_IDX,
                                pred_cat_day >= SOLID_IDX)
        fun_rates = _confusion_rates(fun_conf)
        solid_rates = _confusion_rates(solid_conf)

        n_days_day = int(len(peaks))
        n_fun_obs = fun_conf["tp"] + fun_conf["fn"]
        n_solid_obs = solid_conf["tp"] + solid_conf["fn"]

        out[buoy] = {
            "n_hours": int(bmask.sum()),
            "n_daytime_hours": int(day_sub.sum()),
            "n_days_daytime": n_days_day,
            "n_fun_plus_days_obs": n_fun_obs,
            "n_solid_plus_days_obs": n_solid_obs,

            # Confusion matrix — FUN+
            "fun_tp_days": fun_conf["tp"],
            "fun_fp_days": fun_conf["fp"],
            "fun_fn_days": fun_conf["fn"],
            "fun_tn_days": fun_conf["tn"],
            "fun_precision": fun_rates["precision"],
            "fun_recall": fun_rates["recall"],
            "fun_fp_rate": fun_rates["fp_rate"],
            "fun_fn_rate": fun_rates["fn_rate"],
            "fun_accuracy": fun_rates["accuracy"],

            # Confusion matrix — SOLID+
            "solid_tp_days": solid_conf["tp"],
            "solid_fp_days": solid_conf["fp"],
            "solid_fn_days": solid_conf["fn"],
            "solid_tn_days": solid_conf["tn"],
            "solid_precision": solid_rates["precision"],
            "solid_recall": solid_rates["recall"],
            "solid_fp_rate": solid_rates["fp_rate"],
            "solid_fn_rate": solid_rates["fn_rate"],
            "solid_accuracy": solid_rates["accuracy"],

            # Continuous
            "tp_mae_all": tp_mae_all,
            "hs_mae_ft": hs_mae_ft,
            "fun_plus_hs_mae_ft": fun_plus_hs_mae_ft,
            "dp_circ_mae": dp_cmae,
        }
    return out


def _agg_across_buoys(per_buoy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sum confusion counts; weighted-mean MAEs by n_hours."""
    if not per_buoy:
        return {}
    total_hr = sum(b["n_hours"] for b in per_buoy.values())
    total_day_hr = sum(b["n_daytime_hours"] for b in per_buoy.values())
    total_days = sum(b["n_days_daytime"] for b in per_buoy.values())

    summed_keys = (
        "fun_tp_days", "fun_fp_days", "fun_fn_days", "fun_tn_days",
        "solid_tp_days", "solid_fp_days", "solid_fn_days", "solid_tn_days",
        "n_fun_plus_days_obs", "n_solid_plus_days_obs",
    )
    meaned_keys = ("tp_mae_all", "fun_plus_hs_mae_ft", "hs_mae_ft",
                   "dp_circ_mae")

    agg: dict[str, Any] = {
        "n_hours": int(total_hr),
        "n_daytime_hours": int(total_day_hr),
        "n_days_daytime": int(total_days),
    }
    for k in summed_keys:
        agg[k] = int(sum(b.get(k, 0) for b in per_buoy.values()))
    for k in meaned_keys:
        num = 0.0
        den = 0
        for b in per_buoy.values():
            v = b.get(k)
            w = b.get("n_hours", 0)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            num += v * w
            den += w
        agg[k] = (num / den) if den > 0 else float("nan")
    _attach_rates(agg)
    return agg


def _attach_rates(summary: dict[str, Any]) -> None:
    """Derive precision/recall/fp_rate/fn_rate/accuracy for both FUN and
    SOLID blocks from the summed TP/FP/FN/TN counts in place."""
    for prefix in ("fun", "solid"):
        c = {k: summary.get(f"{prefix}_{k}_days", 0)
             for k in ("tp", "fp", "fn", "tn")}
        r = _confusion_rates(c)
        summary[f"{prefix}_precision"] = r["precision"]
        summary[f"{prefix}_recall"]    = r["recall"]
        summary[f"{prefix}_fp_rate"]   = r["fp_rate"]
        summary[f"{prefix}_fn_rate"]   = r["fn_rate"]
        summary[f"{prefix}_accuracy"]  = r["accuracy"]


# ─── Per-fold fit/predict → per-buoy metrics ─────────────────────────────


def _run_all_folds(df: pd.DataFrame, scope: str, n_folds: int = 4
                   ) -> dict[str, Any]:
    """Re-fit every candidate on every fold and accumulate per-buoy surf
    metrics, scoped to `buoys_for_scope(scope)`."""
    allowed = set(buoys_for_scope(scope))
    print(f"[surf_metrics] scope={scope} buoys={sorted(allowed)}")

    # Pre-compute daytime windows over the full frame once — cheap vs. fold-loop.
    print(f"[surf_metrics] pre-computing daytime windows …")
    daytime_lookup = _build_daytime_lookup(df)

    folds = kfold_month_split(df, n_folds=n_folds)
    per_model_folds: dict[str, list[dict[str, Any]]] = {
        n: [] for n in _candidates()
    }
    fold_meta = [m for _, _, m in folds]

    for fold_i, (train, test, meta) in enumerate(folds):
        print(f"[surf_metrics] ─── fold {fold_i} "
              f"(train={meta['train_rows']:,} test={meta['test_rows']:,}) ───")
        y_train = train[[TARGET_HS, TARGET_TP, TARGET_SIN, TARGET_COS]].copy()
        obs_for_persistence = train[[
            "buoy_id", "valid_utc", "obs_hs_m", "obs_tp_s", "obs_dp_deg"
        ]].copy()

        for name, model in _candidates().items():
            mscope = MODEL_SCOPES[name]
            t0 = time.time()
            try:
                _fit_one(model, name, train, y_train, obs_for_persistence)
                mask = _scope_mask(test, mscope)
                X_test = test[mask].reset_index(drop=True)
                # Further restrict to the user-requested scope's buoys.
                X_test = X_test[X_test["buoy_id"].astype(str).isin(allowed)] \
                    .reset_index(drop=True)
                if len(X_test) == 0:
                    per_model_folds[name].append({
                        "fold": fold_i, "error":
                        f"no rows for scope={scope} ∩ model_scope={mscope}"})
                    continue
                preds = model.predict(X_test)
                per_buoy = _per_buoy_metrics(
                    X_test,
                    preds["pred_hs_m"].to_numpy(),
                    preds["pred_tp_s"].to_numpy(),
                    preds["pred_dp_deg"].to_numpy(),
                    daytime_lookup,
                    allowed,
                )
                summary = _agg_across_buoys(per_buoy)
                per_model_folds[name].append({
                    "fold": fold_i,
                    "scope": scope,
                    "model_scope": mscope,
                    "per_buoy": per_buoy,
                    "summary": summary,
                })
                print(
                    f"[surf_metrics]   {name:<16} "
                    f"FUN FP={summary.get('fun_fp_days', 0):>3} "
                    f"FN={summary.get('fun_fn_days', 0):>3} "
                    f"R={summary.get('fun_recall', float('nan')):.2f}  "
                    f"SOLID FP={summary.get('solid_fp_days', 0):>3} "
                    f"FN={summary.get('solid_fn_days', 0):>3} "
                    f"R={summary.get('solid_recall', float('nan')):.2f}  "
                    f"tpMAE={summary.get('tp_mae_all', float('nan')):.2f}s  "
                    f"fpHsMAE={summary.get('fun_plus_hs_mae_ft', float('nan')):.2f}ft  "
                    f"({time.time()-t0:.1f}s)")
            except Exception:
                traceback.print_exc()
                per_model_folds[name].append({
                    "fold": fold_i, "error": traceback.format_exc()})

    return {"fold_meta": fold_meta, "per_model_folds": per_model_folds}


# ─── Cross-fold rollup ────────────────────────────────────────────────────


def _agg_buoy_over_folds(folds_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse one buoy's per-fold dicts into a single dict.
    Confusion counts sum, MAEs weighted-mean by n_hours."""
    total_hr = sum(m.get("n_hours", 0) for m in folds_metrics)
    total_day_hr = sum(m.get("n_daytime_hours", 0) for m in folds_metrics)
    total_days = sum(m.get("n_days_daytime", 0) for m in folds_metrics)
    summed_keys = (
        "fun_tp_days", "fun_fp_days", "fun_fn_days", "fun_tn_days",
        "solid_tp_days", "solid_fp_days", "solid_fn_days", "solid_tn_days",
        "n_fun_plus_days_obs", "n_solid_plus_days_obs",
    )
    meaned_keys = ("tp_mae_all", "fun_plus_hs_mae_ft", "hs_mae_ft",
                   "dp_circ_mae")
    agg: dict[str, Any] = {
        "n_hours": int(total_hr),
        "n_daytime_hours": int(total_day_hr),
        "n_days_daytime": int(total_days),
    }
    for k in summed_keys:
        agg[k] = int(sum(m.get(k, 0) for m in folds_metrics))
    for k in meaned_keys:
        num = 0.0
        den = 0
        for m in folds_metrics:
            v = m.get(k)
            w = m.get("n_hours", 0)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            num += v * w
            den += w
        agg[k] = (num / den) if den > 0 else float("nan")
    _attach_rates(agg)
    return agg


def _roll_up_per_model(per_model_folds: dict[str, list[dict[str, Any]]]
                       ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, folds in per_model_folds.items():
        ok = [f for f in folds if "error" not in f]
        if not ok:
            out[name] = {"error": "all_folds_failed",
                         "errors": [f.get("error") for f in folds]}
            continue
        combined_per_buoy: dict[str, list[dict[str, Any]]] = {}
        for f in ok:
            for buoy, m in f["per_buoy"].items():
                combined_per_buoy.setdefault(buoy, []).append(m)
        per_buoy_rolled = {
            buoy: _agg_buoy_over_folds(ms)
            for buoy, ms in combined_per_buoy.items()
        }
        summary = _agg_across_buoys(per_buoy_rolled)
        out[name] = {
            "scope": ok[0].get("scope"),
            "model_scope": ok[0].get("model_scope"),
            "per_buoy": per_buoy_rolled,
            "summary": summary,
            "n_folds": len(ok),
        }
    return out


# ─── Composite scoring ────────────────────────────────────────────────────


def _general_swell_score(rolled: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Normalize Hs MAE / Tp MAE / Dp circ-MAE against `mean` baseline,
    then average. 1.0 ≡ mean baseline, 0 = perfect, >1 = worse."""
    ref = (rolled.get("mean") or {}).get("summary") or {}
    ref_hs = ref.get("hs_mae_ft") or float("nan")
    ref_tp = ref.get("tp_mae_all") or float("nan")
    ref_dp = ref.get("dp_circ_mae") or float("nan")

    out: dict[str, float] = {}
    for name, m in rolled.items():
        s = m.get("summary") or {}
        hs = s.get("hs_mae_ft", float("nan"))
        tp = s.get("tp_mae_all", float("nan"))
        dp = s.get("dp_circ_mae", float("nan"))
        terms = []
        if ref_hs and not math.isnan(ref_hs) and not math.isnan(hs):
            terms.append(hs / ref_hs)
        if ref_tp and not math.isnan(ref_tp) and not math.isnan(tp):
            terms.append(tp / ref_tp)
        if ref_dp and not math.isnan(ref_dp) and not math.isnan(dp):
            terms.append(dp / ref_dp)
        out[name] = float(np.mean(terms)) if terms else float("nan")
    return out


def _composite_rank_score(rolled: dict[str, dict[str, Any]],
                          general: dict[str, float]) -> dict[str, float]:
    """Min-max normalize each metric across models to [0, 1] (lower raw =
    lower normalized), then weighted-sum per COMPOSITE_WEIGHTS. Lower =
    better overall."""
    metrics = list(COMPOSITE_WEIGHTS.keys())
    rows: dict[str, dict[str, float]] = {}
    for name, m in rolled.items():
        s = m.get("summary") or {}
        rows[name] = {
            "fun_fn_days":         s.get("fun_fn_days", float("nan")),
            "fun_fp_days":         s.get("fun_fp_days", float("nan")),
            "solid_fn_days":       s.get("solid_fn_days", float("nan")),
            "solid_fp_days":       s.get("solid_fp_days", float("nan")),
            "tp_mae_all":          s.get("tp_mae_all", float("nan")),
            "fun_plus_hs_mae_ft":  s.get("fun_plus_hs_mae_ft", float("nan")),
            "general_swell_score": general.get(name, float("nan")),
        }

    norm: dict[str, dict[str, float]] = {n: {} for n in rows}
    for k in metrics:
        vals = [rows[n][k] for n in rows
                if not (isinstance(rows[n][k], float)
                        and math.isnan(rows[n][k]))]
        if not vals:
            for n in rows:
                norm[n][k] = float("nan")
            continue
        lo, hi = min(vals), max(vals)
        for n in rows:
            v = rows[n][k]
            if isinstance(v, float) and math.isnan(v):
                norm[n][k] = 1.0   # worst
            elif hi == lo:
                norm[n][k] = 0.0
            else:
                norm[n][k] = (v - lo) / (hi - lo)

    total_w = sum(COMPOSITE_WEIGHTS.values())
    out: dict[str, float] = {}
    for n in rows:
        acc = 0.0
        for k, w in COMPOSITE_WEIGHTS.items():
            acc += w * norm[n][k]
        out[n] = acc / total_w
    return out


# ─── Rendering ────────────────────────────────────────────────────────────


def _fmt(v: Any, spec: str) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _render_console(rolled: dict[str, dict[str, Any]],
                    general: dict[str, float],
                    composite: dict[str, float],
                    scope: str) -> str:
    lines: list[str] = []
    lines.append("=" * 150)
    lines.append(
        f"CSC surfer-relevant leaderboard [scope={scope}] "
        "(confusion counts in days; MAEs lower=better; ranked by composite)")
    lines.append("=" * 150)
    hdr = (f"{'rank':>4}  {'model':<16}  {'scope':<5}  "
           f"{'FUN FP':>6}  {'FUN FN':>6}  {'FUN rec':>7}  "
           f"{'SOL FP':>6}  {'SOL FN':>6}  {'SOL rec':>7}  "
           f"{'Tp MAE(s)':>10}  {'FUN+ HsMAE(ft)':>15}  "
           f"{'gen':>6}  {'composite':>10}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    ranked = sorted(composite.items(), key=lambda kv: kv[1])
    for rank, (name, cscore) in enumerate(ranked, 1):
        m = rolled.get(name, {})
        s = m.get("summary") or {}
        lines.append(
            f"{rank:>4}  {name:<16}  {(m.get('model_scope') or '—'):<5}  "
            f"{_fmt(s.get('fun_fp_days'), 'd'):>6}  "
            f"{_fmt(s.get('fun_fn_days'), 'd'):>6}  "
            f"{_fmt(s.get('fun_recall'), '.2f'):>7}  "
            f"{_fmt(s.get('solid_fp_days'), 'd'):>6}  "
            f"{_fmt(s.get('solid_fn_days'), 'd'):>6}  "
            f"{_fmt(s.get('solid_recall'), '.2f'):>7}  "
            f"{_fmt(s.get('tp_mae_all'), '.3f'):>10}  "
            f"{_fmt(s.get('fun_plus_hs_mae_ft'), '.3f'):>15}  "
            f"{_fmt(general.get(name), '.3f'):>6}  "
            f"{_fmt(cscore, '.3f'):>10}"
        )
    return "\n".join(lines)


def _render_markdown(rolled: dict[str, dict[str, Any]],
                     general: dict[str, float],
                     composite: dict[str, float],
                     artifact_dir: Path,
                     generated_at: str,
                     scope: str) -> str:
    lines: list[str] = []
    lines.append(f"# CSC surfer-relevant leaderboard — scope `{scope}`")
    lines.append("")
    lines.append(f"- Generated: `{generated_at}`")
    lines.append(f"- Artifact: `{artifact_dir}`")
    lines.append(f"- Scope: `{scope}` (buoys: "
                 + ", ".join(f"`{b}`" for b in buoys_for_scope(scope)) + ")")
    lines.append("- Confusion counts are **days** inside each buoy's local "
                 "sunrise→sunset window (peak daytime hour per day).")
    lines.append("- `Tp MAE` and `Hs MAE FUN+` are computed across all rows "
                 "(24 h, no daytime filter).")
    lines.append("- Lower is better on every column; ranking uses the "
                 "composite score (weighted sum of min-max-normalized columns).")
    lines.append("- Metric definitions: see `csc/docs/metrics.md`.")
    lines.append("")
    lines.append(
        "| rank | model | scope | days FUN-FP | FUN-FN | FUN recall | "
        "SOLID+-FP | SOLID+-FN | SOLID+ recall | Tp MAE (s) | "
        "Hs MAE FUN+ (ft) | general score | composite |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ranked = sorted(composite.items(), key=lambda kv: kv[1])
    for rank, (name, cscore) in enumerate(ranked, 1):
        m = rolled.get(name, {})
        s = m.get("summary") or {}
        lines.append(
            f"| {rank} | `{name}` | {m.get('model_scope') or '—'} | "
            f"{_fmt(s.get('fun_fp_days'), 'd')} | "
            f"{_fmt(s.get('fun_fn_days'), 'd')} | "
            f"{_fmt(s.get('fun_recall'), '.3f')} | "
            f"{_fmt(s.get('solid_fp_days'), 'd')} | "
            f"{_fmt(s.get('solid_fn_days'), 'd')} | "
            f"{_fmt(s.get('solid_recall'), '.3f')} | "
            f"{_fmt(s.get('tp_mae_all'), '.3f')} | "
            f"{_fmt(s.get('fun_plus_hs_mae_ft'), '.3f')} | "
            f"{_fmt(general.get(name), '.3f')} | "
            f"{_fmt(cscore, '.3f')} |"
        )
    lines.append("")
    lines.append("## Metric weights (composite)")
    lines.append("")
    lines.append("| metric | weight |")
    lines.append("|---|---:|")
    for k, w in COMPOSITE_WEIGHTS.items():
        lines.append(f"| `{k}` | {w} |")
    lines.append("")
    lines.append("## Per-buoy breakdown (best-ranked model)")
    lines.append("")
    if ranked:
        best = ranked[0][0]
        best_per_buoy = (rolled.get(best) or {}).get("per_buoy") or {}
        lines.append(f"Model `{best}` — counts summed across all folds.")
        lines.append("")
        lines.append(
            "| buoy | n_days | FUN obs | FUN TP | FUN FP | FUN FN | "
            "FUN recall | SOLID obs | SOLID TP | SOLID FP | SOLID FN | "
            "SOLID recall | Tp MAE (s) | Hs MAE FUN+ (ft) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for buoy in sorted(best_per_buoy):
            b = best_per_buoy[buoy]
            lines.append(
                f"| {buoy} | {b['n_days_daytime']} | "
                f"{b['n_fun_plus_days_obs']} | "
                f"{b['fun_tp_days']} | {b['fun_fp_days']} | "
                f"{b['fun_fn_days']} | "
                f"{_fmt(b.get('fun_recall'), '.3f')} | "
                f"{b['n_solid_plus_days_obs']} | "
                f"{b['solid_tp_days']} | {b['solid_fp_days']} | "
                f"{b['solid_fn_days']} | "
                f"{_fmt(b.get('solid_recall'), '.3f')} | "
                f"{_fmt(b['tp_mae_all'], '.3f')} | "
                f"{_fmt(b['fun_plus_hs_mae_ft'], '.3f')} |"
            )
    lines.append("")
    return "\n".join(lines)


def _render_per_buoy_percentages(rolled: dict[str, dict[str, Any]],
                                  generated_at: str,
                                  scope: str) -> str:
    lines: list[str] = []
    lines.append(f"# CSC per-buoy percentages — scope `{scope}`")
    lines.append("")
    lines.append(f"- Generated: `{generated_at}`")
    lines.append("- Rows: every (buoy, model) pair across all folds.")
    lines.append("- Confusion counts are DAYS in the local sunrise→sunset "
                 "window (peak hour per day).")
    lines.append("- `precision = TP/(TP+FP)`  `recall = TP/(TP+FN)`")
    lines.append("")
    lines.append(
        "| buoy | model | n_days | FUN obs | FUN TP | FUN FP | FUN FN | "
        "FUN prec | FUN recall | SOLID obs | SOLID TP | SOLID FP | "
        "SOLID FN | SOLID prec | SOLID recall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(rolled):
        per_buoy = (rolled.get(name) or {}).get("per_buoy") or {}
        for buoy in sorted(per_buoy):
            b = per_buoy[buoy]
            lines.append(
                f"| {buoy} | `{name}` | {b.get('n_days_daytime', 0)} | "
                f"{b.get('n_fun_plus_days_obs', 0)} | "
                f"{b.get('fun_tp_days', 0)} | {b.get('fun_fp_days', 0)} | "
                f"{b.get('fun_fn_days', 0)} | "
                f"{_fmt(b.get('fun_precision'), '.3f')} | "
                f"{_fmt(b.get('fun_recall'), '.3f')} | "
                f"{b.get('n_solid_plus_days_obs', 0)} | "
                f"{b.get('solid_tp_days', 0)} | {b.get('solid_fp_days', 0)} | "
                f"{b.get('solid_fn_days', 0)} | "
                f"{_fmt(b.get('solid_precision'), '.3f')} | "
                f"{_fmt(b.get('solid_recall'), '.3f')} |"
            )
    lines.append("")
    return "\n".join(lines)


# ─── Main entry point ────────────────────────────────────────────────────


def run(artifact_dir: Path | None = None,
        n_folds: int = 4,
        scope: str = "east") -> dict[str, Any]:
    """Compute surf-relevant metrics for a given scope and write outputs.

    East scope writes to the canonical filenames (surf_metrics.json,
    model-comparison.md, per-buoy-percentages.md). Non-east writes to
    scope-suffixed files so West never surfaces on the dashboard.
    """
    artifact_dir = (artifact_dir or (CSC_MODELS_DIR / "current")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"[surf_metrics] building training frame …")
    df = build_training_frame()
    if df.empty:
        raise SystemExit(
            "No training frame — run `python -m csc.backfill --full` first.")
    df = add_engineered(df)
    drop_cols = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
        "obs_hs_m", "obs_tp_s", "obs_dp_deg",
    ]
    before = len(df)
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    print(f"[surf_metrics] {before} → {len(df)} rows after core-input dropna")

    # Restrict the whole frame to the requested scope's buoys up front so
    # fold splits and training only see in-scope data.
    allowed = set(buoys_for_scope(scope))
    before_scope = len(df)
    df = df[df["buoy_id"].astype(str).isin(allowed)].reset_index(drop=True)
    print(f"[surf_metrics] {before_scope} → {len(df)} rows after scope={scope} filter")

    fold_dump = _run_all_folds(df, scope=scope, n_folds=n_folds)
    rolled = _roll_up_per_model(fold_dump["per_model_folds"])
    general = _general_swell_score(rolled)
    composite = _composite_rank_score(rolled, general)

    generated_at = datetime.now(timezone.utc).isoformat()
    for name in rolled:
        rolled[name]["general_swell_score"] = general.get(name, float("nan"))
        rolled[name]["composite_rank_score"] = composite.get(name, float("nan"))

    ranked = sorted(composite.items(), key=lambda kv: kv[1])

    payload = {
        "generated_at": generated_at,
        "artifact_dir": str(artifact_dir),
        "scope": scope,
        "scope_buoys": sorted(allowed),
        "n_folds": n_folds,
        "weights": COMPOSITE_WEIGHTS,
        "fold_meta": fold_dump["fold_meta"],
        "ranked": [{"rank": i + 1, "model": n,
                    "composite_rank_score": composite[n]}
                   for i, (n, _) in enumerate(ranked)],
        "models": rolled,
    }

    # Scope-aware filenames — east keeps the canonical names so the
    # dashboard / existing consumers don't need to change.
    suffix = "" if scope == "east" else f"_{scope}"
    json_name = f"surf_metrics{suffix}.json"
    md_name = f"model-comparison{suffix}.md"
    pbp_name = f"per-buoy-percentages{suffix}.md"

    out_path = artifact_dir / json_name
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[surf_metrics] wrote {out_path}")

    docs_dir = Path(__file__).resolve().parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md = _render_markdown(rolled, general, composite, artifact_dir,
                          generated_at, scope)
    (docs_dir / md_name).write_text(md)
    print(f"[surf_metrics] wrote {docs_dir / md_name}")

    pbp = _render_per_buoy_percentages(rolled, generated_at, scope)
    (docs_dir / pbp_name).write_text(pbp)
    print(f"[surf_metrics] wrote {docs_dir / pbp_name}")

    console = _render_console(rolled, general, composite, scope)
    print()
    print(console)
    print()
    print(f"Top 3 by composite surfer-relevance (scope={scope}):")
    for i, (name, cscore) in enumerate(ranked[:3], 1):
        print(f"  {i}. {name}   composite={cscore:.3f}")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="csc.surf_metrics")
    ap.add_argument("--out", type=Path, default=None,
                    help="Artifact dir (default: .csc_models/current)")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--scope", type=str, default="east",
                    choices=["east", "west"],
                    help="CSC scope (default: east; west writes to "
                         "scope-suffixed output files)")
    args = ap.parse_args(argv)
    try:
        run(args.out, n_folds=args.n_folds, scope=args.scope)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
