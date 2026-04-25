"""CSC2 — daily live-eval pass.

For every model under .csc2_models/east/, re-runs inference on cycles
that POST-DATE the model's training run and compares predictions to
buoy obs that have since landed. Writes one row per (model, eval_date)
into .csc2_data/live_eval/<model_name>.parquet so the rolling-skill
trend is visible over time.

Why on-disk: separates "training-time holdout skill" (saved in each
model's meta.json, never changes) from "live skill on truly-unseen
data" (this file, grows daily). The composite skill in
csc2.registry stays training-holdout-based for stability; the live
column is informational.

Run manually:
    python -m csc2.eval_live
Run via launchd: see com.colesurfs.csc2-eval.plist (daily 04:30 ET).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone as dtz
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csc2.schema import CSC2_DATA_DIR, CSC2_MODELS_DIR, LOGS_DIR  # noqa: E402
from csc2.registry import list_models  # noqa: E402
from csc2.train import (  # noqa: E402
    build_paired_dataset, add_features, make_target_columns,
    metric_set, predict_baseline,
)
from csc2.predict import _load_baseline, _load_ml, _ml_predict, detect_arch  # noqa: E402

LIVE_EVAL_DIR = CSC2_DATA_DIR / "live_eval"


def _f1(sens, ppv):
    if sens is None or ppv is None or (sens + ppv) <= 0:
        return None
    return 2.0 * sens * ppv / (sens + ppv)


def _flatten_metrics(metrics_block: dict) -> dict:
    """Flatten the metric_set output for a single model into a single row."""
    row: dict = {}
    for var in ("sw1_height_ft", "sw1_period_s", "sw1_direction_deg",
                "sw2_height_ft", "sw2_period_s", "sw2_direction_deg"):
        b = metrics_block.get(var) or {}
        row[f"{var}_mae"]  = b.get("mae")
        row[f"{var}_rmse"] = b.get("rmse")
        row[f"{var}_bias"] = b.get("bias")
        row[f"{var}_n"]    = b.get("n")
    surfer = (metrics_block.get("surfer") or {}).get("FUN_OR_BETTER") or {}
    row["surfer_FOB_sens"] = surfer.get("sens")
    row["surfer_FOB_ppv"]  = surfer.get("ppv")
    row["surfer_FOB_F1"]   = _f1(surfer.get("sens"), surfer.get("ppv"))
    return row


def eval_one_model(model_dir: Path, paired: pd.DataFrame,
                   trained_ts: float) -> dict | None:
    """Filter paired data to cycles after model.trained_ts, run inference,
    return the flattened metric row. Returns None if no live cycles exist
    (model is too fresh — its training date hasn't been crossed yet)."""
    if paired.empty:
        return None
    cutoff = pd.Timestamp(trained_ts, unit="s", tz="UTC")
    df = paired.copy()
    df["_cycle_dt"] = pd.to_datetime(df["cycle_utc"], format="%Y%m%dT%HZ",
                                      utc=True, errors="coerce")
    df = df[df["_cycle_dt"] > cutoff].copy()
    if df.empty:
        return None

    df = make_target_columns(add_features(df))

    arch = detect_arch(model_dir)
    if arch == "baseline":
        bias = _load_baseline(model_dir)
        pred = predict_baseline(df, bias)
    else:
        boosters = _load_ml(model_dir)
        pred = _ml_predict(df, boosters)

    metrics = metric_set(df, pred, "live_eval")
    flat = _flatten_metrics(metrics)
    flat["n_live_rows"]   = int(len(df))
    flat["n_live_cycles"] = int(df["cycle_utc"].nunique())
    return flat


def append_or_update(out_path: Path, new_row: dict, eval_date: str) -> int:
    """Append a row, or replace any existing row with the same eval_date.
    Returns the resulting total number of rows on disk."""
    new_row = {**new_row, "eval_date_utc": eval_date}
    new_df = pd.DataFrame([new_row])
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
        except Exception:
            existing = pd.DataFrame()
        if "eval_date_utc" in existing.columns:
            existing = existing[existing["eval_date_utc"] != eval_date]
        df_out = pd.concat([existing, new_df], ignore_index=True)
    else:
        df_out = new_df
    df_out = df_out.sort_values("eval_date_utc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False, compression="snappy")
    return len(df_out)


def main() -> int:
    LIVE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    eval_date = datetime.now(dtz.utc).strftime("%Y-%m-%d")

    print(f"[eval_live] eval_date={eval_date}, building paired east-pool dataset…",
          flush=True)
    paired = build_paired_dataset("east")
    if paired.empty:
        print("[eval_live] no paired data on disk; aborting", flush=True)
        return 1
    print(f"[eval_live] paired: {len(paired):,} rows, "
          f"{paired['cycle_utc'].nunique()} cycles", flush=True)

    models = list_models("east")
    print(f"[eval_live] {len(models)} models on disk", flush=True)
    n_evaluated = 0
    n_skipped = 0
    n_errors = 0
    for m in models:
        name = m["name"]
        try:
            row = eval_one_model(Path(m["dir"]), paired, m["trained_ts"])
        except Exception as e:
            print(f"  {name}: ERROR {type(e).__name__}: {e}", flush=True)
            n_errors += 1
            continue
        if row is None:
            print(f"  {name}: no live cycles (skipped)", flush=True)
            n_skipped += 1
            continue
        out_path = LIVE_EVAL_DIR / f"{name}.parquet"
        n_total = append_or_update(out_path, row, eval_date)
        sw1_mae = row.get("sw1_height_ft_mae")
        f1 = row.get("surfer_FOB_F1")
        sw1_str = f"{sw1_mae:.3f}" if sw1_mae is not None else "—"
        f1_str = f"{f1:.3f}" if f1 is not None else "—"
        print(f"  {name}: live_cycles={row['n_live_cycles']:>3} "
              f"rows={row['n_live_rows']:>5} sw1_h_mae={sw1_str} "
              f"FOB_F1={f1_str}  → shard rows={n_total}", flush=True)
        n_evaluated += 1

    elapsed = time.monotonic() - t0
    line = (f"[{datetime.now(dtz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
            f"eval_live  evaluated={n_evaluated} skipped={n_skipped} "
            f"errors={n_errors} elapsed={elapsed:.1f}s\n")
    with (LOGS_DIR / "eval_live.log").open("a") as f:
        f.write(line)
    print(line.rstrip(), flush=True)
    return 0 if n_errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
