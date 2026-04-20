"""CSC FUN+ model variant — LightGBM trained to prefer accuracy when the
true swell is FUN or better (surfable) on the colesurfs categorization scale.

Design choice (approach A from the plan): a standard LightGBM regressor
trained with per-row `sample_weight` — rows whose *observed* (height, period)
pair classifies as FUN or better get weight `fun_plus_weight` (default 3.0);
all other rows get weight 1.0. This is a one-knob tweak, keeps a single
globally-usable model, and targets the distribution shift we care about
without needing a two-stage ensemble.

Also ships FUN+-specific evaluation of a trained bakeoff artifact:

    python -m csc.funplus --evaluate .csc_models/<ts>_v1

which loads the same 80/20 hold-out `csc.experiment.run_bakeoff` uses,
computes FUN+ / non-FUN+ MAE/bias for every candidate (including funplus),
and writes a `fun_plus_report.md` + `fun_plus_report.json` next to the
artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from csc.evaluate import bias as eval_bias, mae as eval_mae
from csc.features import add_engineered, categorical_columns, feature_columns
from csc.models import (
    TARGET_COS, TARGET_HS, TARGET_SIN, TARGET_TP, reconstruct_dp, split_targets,
)
from swell_rules import CATEGORIES, load_bands


M_PER_FT = 0.3048
FT_PER_M = 1.0 / M_PER_FT
FUN_PLUS = {"FUN", "SOLID", "FIRING", "HECTIC", "MONSTRO"}
FUN_PLUS_IDX = np.array([CATEGORIES.index(c) for c in FUN_PLUS], dtype=int)


# ─── FUN+ classification ─────────────────────────────────────────────────


def _category_index(height_ft: float, period_s: float) -> int:
    """Return the CATEGORIES index for one (height_ft, period_s) pair.

    Re-implemented inline (matching `swell_rules.categorize`) so we can
    vectorize on arrays without per-row Python overhead. Returns 0 (FLAT)
    for missing or non-finite inputs.
    """
    if height_ft is None or period_s is None:
        return 0
    if not (np.isfinite(height_ft) and np.isfinite(period_s)):
        return 0
    bands = load_bands()
    band = bands[-1]
    for b in bands:
        ub = b["period_ub"]
        if ub is None or period_s < ub:
            band = b
            break
    rules = band["rules"]
    last_valid_idx = 0
    for i, cat in enumerate(CATEGORIES):
        rule = rules.get(cat, "never")
        if rule == "never":
            continue
        last_valid_idx = i
        if rule == "always":
            break
        if isinstance(rule, dict):
            if height_ft >= rule["gte"]:
                break
        elif isinstance(rule, float):
            if height_ft < rule:
                break
    return last_valid_idx


def is_fun_plus(height_ft: float, period_s: float) -> bool:
    """True iff (height_ft, period_s) classifies as FUN/SOLID/FIRING/HECTIC/MONSTRO."""
    return _category_index(height_ft, period_s) >= CATEGORIES.index("FUN")


def category_index_array(h_ft: np.ndarray, p_s: np.ndarray) -> np.ndarray:
    """Vectorized category-index lookup. Uses the TOML bands directly so
    we only do one pass per band rather than N Python calls. Returns int
    array with value 0 (FLAT) for missing/non-finite inputs.
    """
    h = np.asarray(h_ft, dtype=float)
    p = np.asarray(p_s, dtype=float)
    out = np.zeros(h.shape, dtype=int)
    bad = ~(np.isfinite(h) & np.isfinite(p))

    bands = load_bands()
    # Assign a band index per row based on period_ub
    band_idx = np.full(h.shape, len(bands) - 1, dtype=int)  # default: catch-all
    remaining = np.ones(h.shape, dtype=bool)
    for i, b in enumerate(bands):
        ub = b["period_ub"]
        if ub is None:
            band_idx = np.where(remaining, i, band_idx)
            break
        mask = remaining & (p < ub)
        band_idx = np.where(mask, i, band_idx)
        remaining = remaining & ~mask

    for i, b in enumerate(bands):
        sel = (band_idx == i) & ~bad
        if not sel.any():
            continue
        rules = b["rules"]
        # Walk categories in order; for each row in this band, find the
        # first category whose rule matches.
        h_b = h[sel]
        assigned = np.full(h_b.shape, -1, dtype=int)
        last_valid = np.zeros(h_b.shape, dtype=int)
        done = np.zeros(h_b.shape, dtype=bool)
        for ci, cat in enumerate(CATEGORIES):
            rule = rules.get(cat, "never")
            if rule == "never":
                continue
            last_valid = np.where(done, last_valid, ci)
            if rule == "always":
                assigned = np.where(done, assigned, ci)
                done = np.ones_like(done)
                break
            if isinstance(rule, dict):
                hit = ~done & (h_b >= rule["gte"])
            else:  # float upper bound
                hit = ~done & (h_b < float(rule))
            assigned = np.where(hit, ci, assigned)
            done = done | hit
        final = np.where(assigned >= 0, assigned, last_valid)
        out[sel] = final

    out[bad] = 0
    return out


def is_fun_plus_array(h_ft: np.ndarray, p_s: np.ndarray) -> np.ndarray:
    """Vectorized bool mask: True where (h_ft, p_s) classifies as FUN+."""
    return category_index_array(h_ft, p_s) >= CATEGORIES.index("FUN")


# ─── Sample-weighted LightGBM ────────────────────────────────────────────


class LGBMFunPlus:
    """LightGBM regressor with up-weighted loss on FUN+ observations.

    API matches LGBMCSC exactly, so it slots into the bakeoff with only
    an additional registry entry. The single extra hyperparameter is
    `fun_plus_weight` (default 3.0). Weights are applied only to the Hs
    and Tp target fits (where the categorization actually depends on
    the observation); the sin/cos direction targets use uniform weights
    since direction has no categorical analogue here.
    """

    name = "funplus"

    def __init__(self, n_estimators: int = 600, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_data_in_leaf: int = 100,
                 fun_plus_weight: float = 3.0):
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf,
            feature_fraction=0.9, verbose=-1,
        )
        self.fun_plus_weight = float(fun_plus_weight)
        self.feature_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.models: dict[str, Any] = {}

    def _Xy(self, X: pd.DataFrame) -> pd.DataFrame:
        use = X[self.feature_cols + self.cat_cols].copy()
        for c in self.cat_cols:
            use[c] = use[c].astype("category")
        return use

    def _row_weights(self, hs_m: np.ndarray, tp_s: np.ndarray) -> np.ndarray:
        h_ft = np.asarray(hs_m, dtype=float) * FT_PER_M
        p_s = np.asarray(tp_s, dtype=float)
        fp = is_fun_plus_array(h_ft, p_s)
        return np.where(fp, self.fun_plus_weight, 1.0).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        import lightgbm as lgb
        self.feature_cols = feature_columns(use_partition_features=True)
        self.cat_cols = categorical_columns()
        Xp = self._Xy(X)
        targets = split_targets(y)
        # Weights are keyed off the *observed* Hs/Tp — present in y.
        hs_obs = y[TARGET_HS].to_numpy()
        tp_obs = y[TARGET_TP].to_numpy()
        weights_all = self._row_weights(hs_obs, tp_obs)

        for tgt, series in targets.items():
            mask = ~series.isna()
            w = weights_all if tgt in ("hs", "tp") else None
            m = lgb.LGBMRegressor(**self.params)
            fit_kwargs = {"categorical_feature": self.cat_cols}
            if w is not None:
                fit_kwargs["sample_weight"] = w[mask]
            m.fit(Xp[mask], series[mask].to_numpy(), **fit_kwargs)
            self.models[tgt] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = self._Xy(X)
        hs = self.models["hs"].predict(Xp)
        tp = self.models["tp"].predict(Xp)
        sp = self.models["sin_dp"].predict(Xp)
        cp = self.models["cos_dp"].predict(Xp)
        dp = reconstruct_dp(sp, cp)
        return pd.DataFrame({
            "pred_hs_m": hs, "pred_tp_s": tp, "pred_dp_deg": dp,
        }, index=X.index)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("funplus")
        joblib.dump({
            "params": self.params,
            "fun_plus_weight": self.fun_plus_weight,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "models": self.models,
        }, path / "model.joblib")

    @classmethod
    def load(cls, path: Path):
        obj = joblib.load(path / "model.joblib")
        m = cls(fun_plus_weight=obj.get("fun_plus_weight", 3.0))
        m.params = obj["params"]
        m.feature_cols = obj["feature_cols"]
        m.cat_cols = obj["cat_cols"]
        m.models = obj["models"]
        return m


# ─── Evaluation (FUN+ / non-FUN+ splits + confusion) ─────────────────────


def _category_for_preds(hs_m: np.ndarray, tp_s: np.ndarray) -> np.ndarray:
    """Map predicted (hs_m, tp_s) into CATEGORIES index array."""
    return category_index_array(np.asarray(hs_m) * FT_PER_M, np.asarray(tp_s))


def _stats_for_subset(obs_hs, pred_hs, mask) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "hs_mae_m": float("nan"), "hs_bias_m": float("nan"),
                "hs_mae_ft": float("nan"), "hs_bias_ft": float("nan")}
    mae_m = eval_mae(obs_hs[mask], pred_hs[mask])
    bias_m = eval_bias(obs_hs[mask], pred_hs[mask])
    return {
        "n": n,
        "hs_mae_m": round(float(mae_m), 4),
        "hs_bias_m": round(float(bias_m), 4),
        "hs_mae_ft": round(float(mae_m * FT_PER_M), 3),
        "hs_bias_ft": round(float(bias_m * FT_PER_M), 3),
    }


def _confusion_matrix(obs_cat: np.ndarray, pred_cat: np.ndarray) -> np.ndarray:
    k = len(CATEGORIES)
    m = np.zeros((k, k), dtype=int)
    for o, p in zip(obs_cat, pred_cat):
        m[int(o), int(p)] += 1
    return m


def evaluate_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Load the 80/20 hold-out that the bakeoff used, run every saved model
    in `artifact_dir` against it, and report FUN+ / non-FUN+ Hs accuracy +
    category confusion + POD/precision for FUN+ detection."""
    # Lazy imports to avoid circulars (csc.experiment imports csc.funplus).
    from csc.data import build_training_frame
    from csc.experiment import time_holdout_split
    from csc.models import load_model

    df = build_training_frame()
    df = add_engineered(df)
    drop_cols = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
        "obs_hs_m", "obs_tp_s", "obs_dp_deg",
    ]
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    _, test, cutoff = time_holdout_split(df, test_frac=0.20)
    print(f"[funplus] test rows = {len(test):,}  cutoff = {cutoff}")

    obs_hs = test["obs_hs_m"].to_numpy()
    obs_tp = test["obs_tp_s"].to_numpy()
    obs_cat = category_index_array(obs_hs * FT_PER_M, obs_tp)
    obs_fp = obs_cat >= CATEGORIES.index("FUN")
    print(f"[funplus] FUN+ fraction of test set = {obs_fp.mean():.3f} "
          f"({obs_fp.sum()}/{len(obs_fp)})")

    # Discover every sibling model dir under the artifact
    model_dirs = sorted(p for p in artifact_dir.iterdir()
                        if p.is_dir() and (p / "kind").exists())

    per_model: dict[str, dict[str, Any]] = {}
    for mdir in model_dirs:
        name = mdir.name
        try:
            m = load_model(mdir) if (mdir / "kind").read_text().strip() != "funplus" \
                else LGBMFunPlus.load(mdir)
        except Exception as e:
            per_model[name] = {"error": repr(e)}
            continue
        preds = m.predict(test)
        pred_hs = preds["pred_hs_m"].to_numpy()
        pred_tp = preds["pred_tp_s"].to_numpy()
        pred_cat = _category_for_preds(pred_hs, pred_tp)
        pred_fp = pred_cat >= CATEGORIES.index("FUN")

        hits = int((obs_fp & pred_fp).sum())
        misses = int((obs_fp & ~pred_fp).sum())
        falarms = int((~obs_fp & pred_fp).sum())
        cneg = int((~obs_fp & ~pred_fp).sum())
        pod = hits / (hits + misses) if (hits + misses) else float("nan")
        prec = hits / (hits + falarms) if (hits + falarms) else float("nan")
        acc = (hits + cneg) / len(obs_fp)
        cat_correct = int((obs_cat == pred_cat).sum())
        cat_acc = cat_correct / len(obs_cat)

        per_model[name] = {
            "global": _stats_for_subset(obs_hs, pred_hs, np.ones_like(obs_fp, dtype=bool)),
            "fun_plus": _stats_for_subset(obs_hs, pred_hs, obs_fp),
            "non_fun_plus": _stats_for_subset(obs_hs, pred_hs, ~obs_fp),
            "fun_plus_detection": {
                "hits": hits, "misses": misses,
                "false_alarms": falarms, "correct_negatives": cneg,
                "pod": round(float(pod), 4),
                "precision": round(float(prec), 4),
                "accuracy": round(float(acc), 4),
            },
            "category_accuracy_7way": round(float(cat_acc), 4),
            "confusion_7x7": _confusion_matrix(obs_cat, pred_cat).tolist(),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir.resolve()),
        "test_rows": int(len(test)),
        "holdout_cutoff": str(cutoff),
        "fun_plus_fraction": round(float(obs_fp.mean()), 4),
        "categories": CATEGORIES,
        "models": per_model,
    }
    (artifact_dir / "fun_plus_report.json").write_text(json.dumps(report, indent=2))
    (artifact_dir / "fun_plus_report.md").write_text(_render_md(report))
    print(f"[funplus] wrote {artifact_dir / 'fun_plus_report.json'}")
    print(f"[funplus] wrote {artifact_dir / 'fun_plus_report.md'}")
    return report


# ─── Reporting ───────────────────────────────────────────────────────────


def _render_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# FUN+ evaluation report")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Artifact:  `{report['artifact_dir']}`")
    lines.append(f"- Hold-out rows: {report['test_rows']:,}")
    lines.append(f"- Hold-out cutoff: {report['holdout_cutoff']}")
    lines.append(f"- FUN+ fraction of hold-out: {report['fun_plus_fraction']:.3f}")
    lines.append("")
    lines.append("## Hs MAE / bias by subset (ft)")
    lines.append("")
    lines.append("| model | n (FUN+) | FUN+ MAE | FUN+ bias | non-FUN+ MAE | non-FUN+ bias | global MAE | global bias |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in report["models"].items():
        if "error" in m:
            lines.append(f"| {name} | — | err | err | err | err | err | err |")
            continue
        fp = m["fun_plus"]; nfp = m["non_fun_plus"]; g = m["global"]
        lines.append(
            f"| {name} | {fp['n']} "
            f"| {fp['hs_mae_ft']:.3f} | {fp['hs_bias_ft']:+.3f} "
            f"| {nfp['hs_mae_ft']:.3f} | {nfp['hs_bias_ft']:+.3f} "
            f"| {g['hs_mae_ft']:.3f} | {g['hs_bias_ft']:+.3f} |"
        )
    lines.append("")
    lines.append("## FUN+ detection (categorical)")
    lines.append("")
    lines.append("| model | POD | precision | accuracy | 7-way cat acc |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in report["models"].items():
        if "error" in m:
            continue
        d = m["fun_plus_detection"]
        lines.append(
            f"| {name} | {d['pod']:.3f} | {d['precision']:.3f} "
            f"| {d['accuracy']:.3f} | {m['category_accuracy_7way']:.3f} |"
        )
    lines.append("")
    lines.append("## 7-way category confusion (rows = observed, cols = predicted)")
    lines.append("")
    cats = report["categories"]
    for name, m in report["models"].items():
        if "error" in m:
            continue
        lines.append(f"### {name}")
        lines.append("")
        header = "| obs \\ pred | " + " | ".join(cats) + " |"
        sep = "|" + "---|" * (len(cats) + 1)
        lines.append(header)
        lines.append(sep)
        for i, row in enumerate(m["confusion_7x7"]):
            lines.append(f"| {cats[i]} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    return "\n".join(lines)


def _print_console_report(report: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"FUN+ evaluation — {report['artifact_dir']}")
    print(f"test rows: {report['test_rows']:,}   "
          f"FUN+ fraction: {report['fun_plus_fraction']:.3f}")
    print("=" * 78)
    hdr = f"{'model':<14}  {'FUN+ MAE':>9}  {'FUN+ bias':>10}  " \
          f"{'nFUN+ MAE':>10}  {'global MAE':>11}  {'POD':>5}  {'prec':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in report["models"].items():
        if "error" in m:
            print(f"{name:<14}  error")
            continue
        fp = m["fun_plus"]; nfp = m["non_fun_plus"]; g = m["global"]
        d = m["fun_plus_detection"]
        print(f"{name:<14}  {fp['hs_mae_ft']:>9.3f}  {fp['hs_bias_ft']:>+10.3f}  "
              f"{nfp['hs_mae_ft']:>10.3f}  {g['hs_mae_ft']:>11.3f}  "
              f"{d['pod']:>5.2f}  {d['precision']:>6.2f}")
    print()


def _verdict(report: dict[str, Any]) -> str:
    models = report["models"]
    if "funplus" not in models or "lgbm" not in models:
        return "FunPlus or lgbm missing from report — cannot produce verdict."
    fp = models["funplus"]; lg = models["lgbm"]
    if "error" in fp or "error" in lg:
        return "One of funplus/lgbm errored — cannot produce verdict."

    def d(a, b):
        return a - b

    fp_fun_mae = fp["fun_plus"]["hs_mae_ft"]
    lg_fun_mae = lg["fun_plus"]["hs_mae_ft"]
    fp_nfp_mae = fp["non_fun_plus"]["hs_mae_ft"]
    lg_nfp_mae = lg["non_fun_plus"]["hs_mae_ft"]
    fp_g_mae = fp["global"]["hs_mae_ft"]
    lg_g_mae = lg["global"]["hs_mae_ft"]
    fp_pod = fp["fun_plus_detection"]["pod"]
    lg_pod = lg["fun_plus_detection"]["pod"]
    fp_prec = fp["fun_plus_detection"]["precision"]
    lg_prec = lg["fun_plus_detection"]["precision"]

    improve_fun = d(lg_fun_mae, fp_fun_mae)  # positive = funplus better
    cost_non = d(fp_nfp_mae, lg_nfp_mae)      # positive = funplus worse
    global_delta = d(fp_g_mae, lg_g_mae)      # positive = funplus worse globally
    pod_delta = fp_pod - lg_pod
    prec_delta = fp_prec - lg_prec

    verdict = []
    verdict.append("─── FUN+ verdict ───")
    if improve_fun > 0.005:
        verdict.append(
            f"FunPlus improves Hs MAE on FUN+ rows by {improve_fun:.3f} ft "
            f"({lg_fun_mae:.3f} → {fp_fun_mae:.3f}). The up-weighted training "
            "biased the model toward capturing bigger-swell events, and the "
            "hold-out confirms the shift."
        )
    elif improve_fun < -0.005:
        verdict.append(
            f"FunPlus regresses on FUN+ rows by {-improve_fun:.3f} ft "
            f"({lg_fun_mae:.3f} → {fp_fun_mae:.3f}) — the weighted objective "
            "moved the solution in the wrong direction on this hold-out. "
            "Either the weight ratio is too aggressive or FUN+ rows are "
            "already easy for LightGBM."
        )
    else:
        verdict.append(
            f"FunPlus ties LightGBM on FUN+ rows (Δ={improve_fun:+.3f} ft). "
            "The weighting had no meaningful effect — most FUN+ rows are "
            "already captured by the global optimum."
        )

    if cost_non > 0.005:
        verdict.append(
            f"The cost on non-FUN+ rows is {cost_non:.3f} ft of additional MAE "
            f"({lg_nfp_mae:.3f} → {fp_nfp_mae:.3f}). Globally this pushes Hs MAE "
            f"from {lg_g_mae:.3f} to {fp_g_mae:.3f} ft. Acceptable for a "
            "surfer-facing model; unacceptable if downstream uses care about "
            "small-swell accuracy."
        )
    else:
        verdict.append(
            f"Non-FUN+ MAE is essentially unchanged ({lg_nfp_mae:.3f} → "
            f"{fp_nfp_mae:.3f} ft; Δ={cost_non:+.3f}). The FUN+ gain came "
            "largely free — global MAE moved by {:.3f} ft.".format(global_delta)
        )

    verdict.append(
        f"Categorical FUN+ detection: POD {lg_pod:.2f} → {fp_pod:.2f} "
        f"(Δ={pod_delta:+.2f}), precision {lg_prec:.2f} → {fp_prec:.2f} "
        f"(Δ={prec_delta:+.2f}). {'FunPlus calls more FUN+ events correctly' if pod_delta > 0.01 else 'FunPlus does not materially change FUN+ recall'}; "
        f"{'precision improved' if prec_delta > 0.01 else 'precision held or dropped — more false alarms possible' if prec_delta < -0.01 else 'precision held roughly constant'}."
    )

    if improve_fun > 0.01 and cost_non < 0.02:
        verdict.append(
            "Recommendation: worth promoting as the new default. The FUN+ "
            "gain exceeds the non-FUN+ cost, and the user-visible behaviour "
            "(surfable-swell accuracy) improves."
        )
    elif improve_fun > 0.005:
        verdict.append(
            "Recommendation: keep funplus as a sibling candidate. It helps "
            "on FUN+ rows but the tradeoff against global accuracy isn't "
            "lopsided enough to auto-promote — leave the decision to the user."
        )
    else:
        verdict.append(
            "Recommendation: keep as a sibling candidate for transparency, "
            "but do not promote. The weighted objective did not meaningfully "
            "shift the FUN+ error; try a higher weight or approach B/C."
        )

    return "\n".join(verdict)


# ─── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="csc.funplus")
    ap.add_argument("--evaluate", type=Path, metavar="ARTIFACT_DIR",
                    help="Directory produced by csc.experiment/csc.train")
    args = ap.parse_args(argv)

    if args.evaluate is None:
        ap.print_help()
        return 2
    if not args.evaluate.exists():
        print(f"[funplus] no such dir: {args.evaluate}", file=sys.stderr)
        return 1

    report = evaluate_artifact(args.evaluate)
    _print_console_report(report)
    print(_verdict(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
