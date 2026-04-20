"""Verification metrics for CSC — the standard wave-model suite.

Given predicted and observed columns (pred_hs_m, pred_tp_s, pred_dp_deg
vs obs_hs_m, obs_tp_s, obs_dp_deg), compute the full metric bundle
used to compare GFS/Euro/CSC on the /csc dashboard: bias, MAE, RMSE,
SI, HH, r, regression slope, ACC, NSE, POD/FAR/CSI at thresholds,
circular MAE, vector correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Scalar metrics ───────────────────────────────────────────────────────

def _clean(obs: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(obs) & np.isfinite(pred)
    return obs[m], pred[m]


def bias(obs, pred) -> float:
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if not o.size: return float("nan")
    return float(np.mean(p - o))


def mae(obs, pred) -> float:
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if not o.size: return float("nan")
    return float(np.mean(np.abs(p - o)))


def rmse(obs, pred) -> float:
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if not o.size: return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)))


def scatter_index(obs, pred) -> float:
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if not o.size or np.mean(o) == 0: return float("nan")
    return float(rmse(o, p) / np.mean(o))


def hh_index(obs, pred) -> float:
    """Hanna-Heinold index."""
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    denom = float(np.sum(p * o))
    if not o.size or denom <= 0: return float("nan")
    return float(np.sqrt(np.sum((p - o) ** 2) / denom))


def pearson_r(obs, pred) -> float:
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if o.size < 3: return float("nan")
    return float(np.corrcoef(o, p)[0, 1])


def regression_slope(obs, pred) -> float:
    """Observed-on-predicted OLS slope — <1 = under-amplify, >1 = over-amplify."""
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if o.size < 3: return float("nan")
    m, _ = np.polyfit(p, o, 1)
    return float(m)


def anomaly_correlation(obs, pred, climatology: float | None = None) -> float:
    """ACC against a provided climatological mean (or overall mean)."""
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if o.size < 3: return float("nan")
    c = np.mean(o) if climatology is None else float(climatology)
    oa = o - c; pa = p - c
    denom = float(np.sqrt(np.sum(oa ** 2)) * np.sqrt(np.sum(pa ** 2)))
    if denom == 0: return float("nan")
    return float(np.sum(oa * pa) / denom)


def nse(obs, pred) -> float:
    """Nash-Sutcliffe efficiency — skill vs climatology (constant-mean)."""
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    if o.size < 3: return float("nan")
    denom = float(np.sum((o - np.mean(o)) ** 2))
    if denom == 0: return float("nan")
    return float(1.0 - np.sum((o - p) ** 2) / denom)


def skill_score(obs, pred, baseline_pred) -> float:
    """SS = 1 − MSE(model) / MSE(baseline). Positive = better than baseline."""
    o1, p1 = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    o2, b2 = _clean(np.asarray(obs, dtype=float), np.asarray(baseline_pred, dtype=float))
    # use the intersection
    n = min(o1.size, o2.size)
    if n < 3: return float("nan")
    mse_m = float(np.mean((np.asarray(pred)[:n] - np.asarray(obs)[:n]) ** 2))
    mse_b = float(np.mean((np.asarray(baseline_pred)[:n] - np.asarray(obs)[:n]) ** 2))
    if mse_b == 0: return float("nan")
    return float(1.0 - mse_m / mse_b)


# ─── Categorical event metrics ────────────────────────────────────────────

def contingency(obs, pred, threshold: float) -> dict[str, int]:
    o, p = _clean(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float))
    oe = o > threshold; pe = p > threshold
    hits  = int(np.sum(oe & pe))
    miss  = int(np.sum(oe & ~pe))
    falma = int(np.sum(~oe & pe))
    cn    = int(np.sum(~oe & ~pe))
    return {"hits": hits, "misses": miss, "false_alarms": falma, "correct_negatives": cn}


def pod(obs, pred, threshold: float) -> float:
    c = contingency(obs, pred, threshold)
    denom = c["hits"] + c["misses"]
    return float("nan") if denom == 0 else c["hits"] / denom


def far(obs, pred, threshold: float) -> float:
    c = contingency(obs, pred, threshold)
    denom = c["hits"] + c["false_alarms"]
    return float("nan") if denom == 0 else c["false_alarms"] / denom


def csi(obs, pred, threshold: float) -> float:
    c = contingency(obs, pred, threshold)
    denom = c["hits"] + c["misses"]+ c["false_alarms"]
    return float("nan") if denom == 0 else c["hits"] / denom


def heidke_skill_score(obs_cats: np.ndarray, pred_cats: np.ndarray,
                       n_categories: int) -> float:
    """Categorical skill vs random-chance baseline."""
    if obs_cats.size == 0: return float("nan")
    m = np.zeros((n_categories, n_categories), dtype=int)
    for o, p in zip(obs_cats, pred_cats):
        if 0 <= o < n_categories and 0 <= p < n_categories:
            m[int(o), int(p)] += 1
    N = m.sum()
    if N == 0: return float("nan")
    on_diag = np.trace(m)
    row_sums = m.sum(axis=1)
    col_sums = m.sum(axis=0)
    expected = float(np.sum(row_sums * col_sums)) / N
    denom = N - expected
    return float("nan") if denom == 0 else (on_diag - expected) / denom


# ─── Direction-specific ───────────────────────────────────────────────────

def _angle_diff(a, b):
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def circular_mae(obs_deg, pred_deg) -> float:
    o = np.asarray(obs_deg, dtype=float); p = np.asarray(pred_deg, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    if not m.any(): return float("nan")
    return float(np.mean(_angle_diff(p[m], o[m])))


def circular_bias(obs_deg, pred_deg) -> float:
    """Signed mean angular error via complex-mean trick."""
    o = np.asarray(obs_deg, dtype=float); p = np.asarray(pred_deg, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    if not m.any(): return float("nan")
    diff = np.deg2rad((p[m] - o[m]) % 360.0)
    mean = np.angle(np.mean(np.exp(1j * diff)))
    return float(np.rad2deg(mean))


def vector_correlation(obs_deg, pred_deg) -> float:
    """Circular (vector) correlation — for unit vectors on the compass."""
    o = np.asarray(obs_deg, dtype=float); p = np.asarray(pred_deg, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    if m.sum() < 3: return float("nan")
    o = np.deg2rad(o[m]); p = np.deg2rad(p[m])
    so, co = np.sin(o).mean(), np.cos(o).mean()
    sp, cp = np.sin(p).mean(), np.cos(p).mean()
    num = (np.sin(o - so) * np.sin(p - sp)).sum() + (np.cos(o - co) * np.cos(p - cp)).sum()
    den = (np.sqrt(((np.sin(o - so) ** 2 + np.cos(o - co) ** 2).sum()) *
                   ((np.sin(p - sp) ** 2 + np.cos(p - cp) ** 2).sum())))
    return float("nan") if den == 0 else float(num / den)


# ─── Top-level rollup ─────────────────────────────────────────────────────

HS_THRESHOLDS_FT = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]


def summarize(df: pd.DataFrame, baseline_hs: np.ndarray | None = None,
              baseline_tp: np.ndarray | None = None) -> dict[str, float]:
    """Compute all metrics on a DataFrame with both obs_* and pred_* cols."""
    d: dict[str, float] = {}
    for var, obs_col, pred_col in [
        ("hs", "obs_hs_m", "pred_hs_m"),
        ("tp", "obs_tp_s", "pred_tp_s"),
    ]:
        obs = df[obs_col].to_numpy()
        pred = df[pred_col].to_numpy()
        d[f"{var}_bias"] = bias(obs, pred)
        d[f"{var}_mae"] = mae(obs, pred)
        d[f"{var}_rmse"] = rmse(obs, pred)
        d[f"{var}_si"] = scatter_index(obs, pred)
        d[f"{var}_hh"] = hh_index(obs, pred)
        d[f"{var}_r"] = pearson_r(obs, pred)
        d[f"{var}_slope"] = regression_slope(obs, pred)
        d[f"{var}_acc"] = anomaly_correlation(obs, pred)
        d[f"{var}_nse"] = nse(obs, pred)

    # direction
    obs_dp = df["obs_dp_deg"].to_numpy()
    pred_dp = df["pred_dp_deg"].to_numpy()
    d["dp_circ_mae"] = circular_mae(obs_dp, pred_dp)
    d["dp_bias"] = circular_bias(obs_dp, pred_dp)
    d["dp_vecr"] = vector_correlation(obs_dp, pred_dp)

    # Hs threshold event skill (meters → convert thresholds from ft)
    obs_hs_m = df["obs_hs_m"].to_numpy()
    pred_hs_m = df["pred_hs_m"].to_numpy()
    for thr_ft in HS_THRESHOLDS_FT:
        thr_m = thr_ft / 3.28084
        d[f"pod_hs_gt_{int(thr_ft)}ft"] = pod(obs_hs_m, pred_hs_m, thr_m)
        d[f"far_hs_gt_{int(thr_ft)}ft"] = far(obs_hs_m, pred_hs_m, thr_m)
        d[f"csi_hs_gt_{int(thr_ft)}ft"] = csi(obs_hs_m, pred_hs_m, thr_m)

    # Skill scores
    if baseline_hs is not None:
        d["hs_skill_vs_baseline"] = skill_score(obs_hs_m, pred_hs_m, baseline_hs)
    if baseline_tp is not None:
        d["tp_skill_vs_baseline"] = skill_score(
            df["obs_tp_s"].to_numpy(), df["pred_tp_s"].to_numpy(), baseline_tp)
    return d
