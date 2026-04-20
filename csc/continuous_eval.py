"""CSC continuous evaluation — weekly rollup across every preserved artifact.

Triggered by com.colesurfs.csc-eval.plist (Mondays at 06:00 local). For each
artifact under .csc_models/* (not just the promoted one), reruns the
surfer-relevant metric suite against the latest month of joined
forecast/observation data and persists a snapshot + a running time series.

Files written:
  .csc_data/weekly_eval/YYYY-MM-DD.json       — per-week rolling snapshot
  .csc_data/weekly_eval/latest.json           — symlink-style copy (overwrite)
  .csc_data/weekly_eval/timeseries.parquet    — append-only per-buoy series

Metric source order:
  1. csc.surf_metrics.evaluate_artifact(artifact_dir, df) if available
     (another agent is writing that module in parallel — it will supply
      the surfer-specific rollup: Hs/Tp/Dp MAE, missed_fun_days,
      false_pos_solid_days).
  2. Fallback: csc.evaluate.summarize over the latest-month joined frame.

Designed to fail soft — every artifact evaluation is wrapped, and the
launchd entry logs to /tmp/csc-eval.log. A poisoned artifact can't crash
the run and prevent the healthy ones from being recorded.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csc.display_filter import is_public_buoy, is_public_variant
from csc.schema import BUOY_IDS, CSC_DATA_DIR, CSC_MODELS_DIR

WEEKLY_DIR = CSC_DATA_DIR / "weekly_eval"
TIMESERIES_PATH = WEEKLY_DIR / "timeseries.parquet"
LATEST_PATH = WEEKLY_DIR / "latest.json"

TIMESERIES_COLS = [
    "eval_date", "model_version", "buoy_id",
    "hs_mae", "tp_mae", "dp_mae",
    "missed_fun_days", "false_pos_solid_days",
]


# ─── Artifact discovery ───────────────────────────────────────────────────

def _list_artifacts() -> list[Path]:
    """Every preserved artifact under .csc_models/ except the `current`
    symlink. Sorted by directory name (version timestamp) ascending."""
    root = Path(CSC_MODELS_DIR)
    if not root.exists():
        return []
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.is_symlink():
            continue
        if d.name.startswith(".") or d.name == "current":
            continue
        if not (d / "manifest.json").exists():
            continue
        out.append(d)
    return out


def _load_model(artifact_dir: Path):
    """Load the winner model for an artifact. Returns None on failure."""
    try:
        manifest = json.loads((artifact_dir / "manifest.json").read_text())
    except Exception:
        return None, None
    winner = manifest.get("winner")
    winner_dir = Path(manifest.get("winner_dir", ""))
    if not winner_dir.is_absolute():
        winner_dir = artifact_dir / (winner or "")
    if not winner_dir.exists():
        return None, manifest
    try:
        from csc.models import load_model
        return load_model(winner_dir), manifest
    except Exception:
        traceback.print_exc()
        return None, manifest


# ─── Evaluation dataset — latest month window ────────────────────────────

def _latest_month_frame() -> pd.DataFrame | None:
    """Build the last-30-day joined forecast/observation frame, with the
    engineered features the models need."""
    try:
        from csc.data import build_training_frame
        from csc.features import add_engineered
    except Exception:
        traceback.print_exc()
        return None
    df = build_training_frame()
    if df is None or df.empty:
        return None
    cutoff = df["valid_utc"].max() - pd.Timedelta(days=30)
    df = df[df["valid_utc"] >= cutoff].copy()
    if df.empty:
        return None
    df = add_engineered(df)
    drop_cols = [
        "gfs_wave_height", "gfs_wave_period", "gfs_wave_direction",
        "euro_wave_height", "euro_wave_period", "euro_wave_direction",
        "obs_hs_m", "obs_tp_s", "obs_dp_deg",
    ]
    df = df.dropna(subset=[c for c in drop_cols if c in df.columns])
    return df.reset_index(drop=True)


def _angle_diff(a, b) -> np.ndarray:
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def _per_buoy_metrics(df: pd.DataFrame, preds: pd.DataFrame) -> dict[str, dict]:
    """Compute Hs/Tp/Dp MAE and simple FUN+/SOLID-day categorical counts
    broken down per buoy. The FUN+/SOLID thresholds are conservative
    defaults — when csc.surf_metrics is available it supersedes these.

    Thresholds (Hs, ft):
      FUN+    ≥ 2.5 ft (~0.76 m)
      SOLID   ≥ 4.0 ft (~1.22 m)
    missed_fun_days        = unique UTC dates with obs FUN+ but pred < FUN+
    false_pos_solid_days   = unique UTC dates with pred SOLID but obs < SOLID
    """
    FUN_M = 2.5 / 3.28084
    SOLID_M = 4.0 / 3.28084
    out: dict[str, dict] = {}

    merged = df[["buoy_id", "valid_utc", "obs_hs_m", "obs_tp_s", "obs_dp_deg"]].copy()
    merged = merged.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    for col in ("pred_hs_m", "pred_tp_s", "pred_dp_deg"):
        if col in preds.columns:
            merged[col] = preds[col].values

    merged["date_utc"] = pd.to_datetime(merged["valid_utc"], utc=True).dt.date

    for buoy_id, g in merged.groupby("buoy_id"):
        n = len(g)
        hs_mae = float(np.mean(np.abs(g["pred_hs_m"] - g["obs_hs_m"]))) if "pred_hs_m" in g else float("nan")
        tp_mae = float(np.mean(np.abs(g["pred_tp_s"] - g["obs_tp_s"]))) if "pred_tp_s" in g else float("nan")
        if "pred_dp_deg" in g:
            dp_mae = float(np.mean(_angle_diff(g["obs_dp_deg"].to_numpy(),
                                               g["pred_dp_deg"].to_numpy())))
        else:
            dp_mae = float("nan")

        missed_fun = 0
        false_pos_solid = 0
        if "pred_hs_m" in g:
            by_day = g.groupby("date_utc").agg(
                obs_max=("obs_hs_m", "max"),
                pred_max=("pred_hs_m", "max"),
            )
            missed_fun = int(((by_day["obs_max"] >= FUN_M) &
                              (by_day["pred_max"] < FUN_M)).sum())
            false_pos_solid = int(((by_day["pred_max"] >= SOLID_M) &
                                   (by_day["obs_max"] < SOLID_M)).sum())
        out[str(buoy_id)] = {
            "n": n,
            "hs_mae": hs_mae,
            "tp_mae": tp_mae,
            "dp_mae": dp_mae,
            "missed_fun_days": missed_fun,
            "false_pos_solid_days": false_pos_solid,
        }
    return out


def _evaluate_one_artifact(artifact_dir: Path,
                           df: pd.DataFrame) -> dict[str, Any] | None:
    """Score one artifact. Returns dict with shape:
      {version, winner, global: {...}, per_buoy: {buoy_id: {...}}, source}
    Uses csc.surf_metrics.evaluate_artifact when available, otherwise
    falls back to csc.evaluate.summarize + per-buoy compute.
    """
    # ── Preferred: surf_metrics ─────────────────────────────────────────
    try:
        from csc import surf_metrics  # type: ignore
        evaluator = getattr(surf_metrics, "evaluate_artifact", None)
    except Exception:
        surf_metrics = None
        evaluator = None

    model, manifest = _load_model(artifact_dir)
    version = (manifest or {}).get("version") or artifact_dir.name
    winner = (manifest or {}).get("winner")

    if evaluator is not None and model is not None:
        try:
            result = evaluator(artifact_dir, df)
            if isinstance(result, dict):
                result.setdefault("version", version)
                result.setdefault("winner", winner)
                result.setdefault("source", "surf_metrics")
                return result
        except Exception:
            traceback.print_exc()

    # ── Fallback: csc.evaluate.summarize + per-buoy ─────────────────────
    if model is None:
        return {"version": version, "winner": winner,
                "error": "model_load_failed", "source": "fallback"}
    try:
        preds = model.predict(df)
    except Exception:
        traceback.print_exc()
        return {"version": version, "winner": winner,
                "error": "predict_failed", "source": "fallback"}

    scored = df[["buoy_id", "valid_utc", "obs_hs_m", "obs_tp_s", "obs_dp_deg"]].copy()
    for c in ("pred_hs_m", "pred_tp_s", "pred_dp_deg"):
        if c in preds.columns:
            scored[c] = preds[c].values

    try:
        from csc.evaluate import summarize
        globals_block = summarize(scored)
    except Exception:
        traceback.print_exc()
        globals_block = {}

    per_buoy = _per_buoy_metrics(df, preds)

    return {
        "version": version,
        "winner": winner,
        "source": "fallback_summarize",
        "n": int(len(scored)),
        "global": globals_block,
        "per_buoy": per_buoy,
    }


# ─── Time-series append ───────────────────────────────────────────────────

def _append_timeseries(eval_date: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows, columns=TIMESERIES_COLS)
    if TIMESERIES_PATH.exists():
        try:
            existing = pd.read_parquet(TIMESERIES_PATH)
            df = pd.concat([existing, new], ignore_index=True)
        except Exception:
            # Corrupt — sidestep rather than lose new data
            corrupt = TIMESERIES_PATH.with_suffix(".parquet.corrupt")
            try:
                TIMESERIES_PATH.rename(corrupt)
            except OSError:
                pass
            df = new
    else:
        df = new
    df = df.drop_duplicates(
        subset=["eval_date", "model_version", "buoy_id"], keep="last",
    )
    TIMESERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TIMESERIES_PATH, index=False, compression="snappy")


def _snapshot_to_timeseries_rows(eval_date: str,
                                 snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_entry in snapshot.get("models", []):
        version = model_entry.get("version")
        per_buoy = model_entry.get("per_buoy") or {}
        for buoy_id, m in per_buoy.items():
            rows.append({
                "eval_date": eval_date,
                "model_version": version,
                "buoy_id": str(buoy_id),
                "hs_mae": float(m.get("hs_mae", float("nan"))),
                "tp_mae": float(m.get("tp_mae", float("nan"))),
                "dp_mae": float(m.get("dp_mae", float("nan"))),
                "missed_fun_days": int(m.get("missed_fun_days", 0) or 0),
                "false_pos_solid_days": int(m.get("false_pos_solid_days", 0) or 0),
            })
    return rows


# ─── Public API ───────────────────────────────────────────────────────────

def run_weekly_eval() -> dict[str, Any]:
    """Score every preserved artifact on the latest month of data and
    persist a snapshot + update the running per-buoy time series."""
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

    eval_dt = datetime.now(timezone.utc)
    eval_date = eval_dt.strftime("%Y-%m-%d")

    df = _latest_month_frame()
    if df is None or df.empty:
        snapshot = {
            "eval_date": eval_date,
            "generated_at": eval_dt.isoformat(),
            "error": "no_eval_data",
            "buoys": BUOY_IDS,
            "models": [],
        }
    else:
        artifacts = _list_artifacts()
        entries: list[dict[str, Any]] = []
        for a in artifacts:
            try:
                res = _evaluate_one_artifact(a, df)
            except Exception:
                traceback.print_exc()
                res = {"version": a.name, "error": "eval_crashed"}
            if res is not None:
                entries.append(res)
        snapshot = {
            "eval_date": eval_date,
            "generated_at": eval_dt.isoformat(),
            "window_days": 30,
            "window_start_utc": str(df["valid_utc"].min()),
            "window_end_utc": str(df["valid_utc"].max()),
            "eval_rows": int(len(df)),
            "buoys": sorted(df["buoy_id"].astype(str).unique().tolist()),
            "n_artifacts": len(artifacts),
            "models": entries,
        }

    (WEEKLY_DIR / f"{eval_date}.json").write_text(
        json.dumps(snapshot, indent=2, default=str))
    LATEST_PATH.write_text(json.dumps(snapshot, indent=2, default=str))

    try:
        rows = _snapshot_to_timeseries_rows(eval_date, snapshot)
        _append_timeseries(eval_date, rows)
    except Exception:
        traceback.print_exc()

    return snapshot


def public_scope(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Filter a snapshot to public (East Coast) scope for dashboard consumers.

    Training/eval runs over the full buoy set so cross-coast signal is
    preserved in the on-disk snapshot. This helper strips:
      - west-only trained variants (via is_public_variant)
      - west-only per-buoy entries (via is_public_buoy)
    Leaves the original snapshot object untouched.
    """
    if not isinstance(snapshot, dict):
        return snapshot
    out = dict(snapshot)
    models = out.get("models") or []
    pub_models: list[dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        version = m.get("version") or ""
        winner = m.get("winner") or ""
        if not (is_public_variant(version) and is_public_variant(winner)):
            continue
        m2 = dict(m)
        pb = m2.get("per_buoy") or {}
        if isinstance(pb, dict):
            m2["per_buoy"] = {bid: v for bid, v in pb.items()
                              if is_public_buoy(bid)}
        pub_models.append(m2)
    out["models"] = pub_models
    buoys = out.get("buoys")
    if isinstance(buoys, list):
        out["buoys"] = [b for b in buoys if is_public_buoy(b)]
    return out


def latest_snapshot() -> dict[str, Any]:
    """Return the latest persisted snapshot (or a structured empty one),
    filtered to public (East Coast) scope."""
    if not LATEST_PATH.exists():
        return {"error": "no_snapshot_yet",
                "hint": "run `python -m csc.continuous_eval`"}
    try:
        snap = json.loads(LATEST_PATH.read_text())
    except Exception:
        return {"error": "snapshot_unreadable"}
    return public_scope(snap)


# ─── CLI entry ────────────────────────────────────────────────────────────

def main() -> int:
    try:
        snap = run_weekly_eval()
    except Exception:
        traceback.print_exc()
        return 2
    n_models = len(snap.get("models", []))
    print(f"[weekly-eval] scored {n_models} artifacts "
          f"(eval_date={snap.get('eval_date')}, "
          f"rows={snap.get('eval_rows')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
