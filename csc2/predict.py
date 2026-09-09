"""CSC2 inference — load a saved baseline-or-ML model and apply it to a
live EURO+GFS cycle to produce dashboard-comparable corrected swell rows.

The predictor mirrors `csc2.train`'s feature engineering exactly, so the
distribution at inference matches training. Both architectures share the
same input schema (lead-aligned EURO + GFS forecast records for one
buoy at one cycle).

Inputs to `predict_for_cycle`:
    model_dir : Path to .csc2_models/east/<name>/
    buoy_id   : str — selects the buoy_one_hot feature
    euro_recs : list of dashboard-format EURO records (waves_cmems output)
    gfs_recs  : list of dashboard-format GFS records (waves output)
    cycle_utc : str like '20260424T12Z' — anchor for lead_hours

Returns: list of dicts, one per (lead_hour) where both EURO and GFS have
data, each with corrected sw1/sw2 (height/period/direction). Compatible
with the dashboard's record schema so it can drop into a CSC2 column.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from csc2.train import (
    _apply_dashboard_fallback_gfs,
    add_features,
    predict_baseline,
    predict_ml,
)

OUTPUT_DASHBOARD_VARS = (
    "sw1_height_ft", "sw1_period_s", "sw1_direction_deg",
    "sw2_height_ft", "sw2_period_s", "sw2_direction_deg",
)


# ---------------------------------------------------------------------------
# Build feature frame from live records
# ---------------------------------------------------------------------------

def _local_to_utc_iso(s: str) -> str:
    """Mirror csc2/logger.py:_local_to_utc — dashboard records carry NY local
    timestamps; we need ISO-Z UTC for joining."""
    try:
        from zoneinfo import ZoneInfo
        t_local = datetime.fromisoformat(s).replace(tzinfo=ZoneInfo("America/New_York"))
        return t_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _records_to_frame(recs: list, buoy_id: str, model: str,
                      cycle_utc: str) -> pd.DataFrame:
    """Flatten dashboard records into the same wide schema the trainer reads
    off the parquet shards. This is what csc2/logger.py would write, just
    constructed in-memory for live cycles."""
    rows = []
    for r in recs:
        valid_utc = _local_to_utc_iso(r.get("time", ""))
        if not valid_utc:
            continue
        try:
            vdt = datetime.strptime(valid_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            cdt = datetime.strptime(cycle_utc, "%Y%m%dT%HZ").replace(tzinfo=timezone.utc)
            lead = int(round((vdt - cdt).total_seconds() / 3600))
        except Exception:
            lead = None
        comps = r.get("components") or []
        c0 = comps[0] if len(comps) > 0 else {}
        c1 = comps[1] if len(comps) > 1 else {}
        rows.append({
            "buoy_id": buoy_id,
            "model": model,
            "cycle_utc": cycle_utc,
            "valid_utc": valid_utc,
            "lead_hours": lead,
            "sw1_height_ft": c0.get("height_ft"),
            "sw1_period_s":  c0.get("period_s"),
            "sw1_direction_deg": c0.get("direction_deg"),
            "sw2_height_ft": c1.get("height_ft"),
            "sw2_period_s":  c1.get("period_s"),
            "sw2_direction_deg": c1.get("direction_deg"),
            "combined_height_m":  r.get("combined_wave_height_m"),
            "combined_period_s":  r.get("combined_wave_period_s"),
            "combined_direction_deg": r.get("combined_wave_direction_deg"),
        })
    return pd.DataFrame(rows)


def _join_models(euro_df: pd.DataFrame, gfs_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["buoy_id", "cycle_utc", "valid_utc", "lead_hours",
            "sw1_height_ft", "sw1_period_s", "sw1_direction_deg",
            "sw2_height_ft", "sw2_period_s", "sw2_direction_deg",
            "combined_height_m", "combined_period_s", "combined_direction_deg"]
    e = euro_df[keep].add_prefix("euro_").rename(columns={
        "euro_buoy_id": "buoy_id", "euro_cycle_utc": "cycle_utc",
        "euro_valid_utc": "valid_utc", "euro_lead_hours": "lead_hours",
    })
    g = gfs_df[keep].add_prefix("gfs_").rename(columns={
        "gfs_buoy_id": "buoy_id", "gfs_cycle_utc": "cycle_utc",
        "gfs_valid_utc": "valid_utc", "gfs_lead_hours": "lead_hours",
    })
    return e.merge(g, on=["buoy_id", "cycle_utc", "valid_utc", "lead_hours"], how="inner")


# ---------------------------------------------------------------------------
# Baseline inference
# ---------------------------------------------------------------------------

def _load_baseline(model_dir: Path) -> dict:
    p = model_dir / "bias.json"
    raw = json.loads(p.read_text())
    bias = {"scalar": {}, "dp": {}}
    for v, tbl in (raw.get("scalar") or {}).items():
        bias["scalar"][v] = {tuple(k.split("|", 1)): float(x) for k, x in tbl.items()}
        bias["scalar"][v] = {(b, int(l)): x for (b, l), x in bias["scalar"][v].items()}
    for sw, tbl in (raw.get("dp") or {}).items():
        d = {}
        for k, sc in tbl.items():
            b, l = k.split("|", 1)
            d[(b, int(l))] = (float(sc[0]), float(sc[1]))
        bias["dp"][sw] = d
    return bias


# Inference itself is csc2.train.predict_baseline / predict_ml — the trainer's
# holdout scoring and live prediction run the identical code path.


# ---------------------------------------------------------------------------
# ML inference
# ---------------------------------------------------------------------------

def _load_ml(model_dir: Path) -> dict:
    import lightgbm as lgb
    feat_cols = json.loads((model_dir / "feature_cols.json").read_text())
    boosters = {}
    for f in sorted(model_dir.glob("booster_*.txt")):
        target = f.name[len("booster_"):-len(".txt")]
        booster = lgb.Booster(model_file=str(f))
        # Guard against silent feature misalignment: a booster saved with a
        # different feature order would produce garbage, not an error.
        bf = booster.feature_name()
        if bf != feat_cols:
            raise ValueError(
                f"{model_dir.name}/{f.name}: booster features do not match "
                f"feature_cols.json ({len(bf)} vs {len(feat_cols)} cols, "
                f"first diff at {next((i for i, (a, b) in enumerate(zip(bf, feat_cols)) if a != b), 'length')})")
        boosters[target] = booster
    return {"feature_cols": feat_cols, "models": boosters}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_arch(model_dir: Path) -> str:
    if (model_dir / "bias.json").exists():
        return "baseline"
    if list(model_dir.glob("booster_*.txt")):
        return "ML"
    raise FileNotFoundError(f"unknown model layout in {model_dir}")


def predict_for_cycle(model_dir: Path, *, buoy_id: str,
                      euro_recs: list, gfs_recs: list,
                      cycle_utc: str) -> list[dict]:
    """End-to-end inference for one (model, buoy, cycle) triple. Returns
    a list of dicts, one per matched valid_utc, each with corrected
    dashboard-shape fields (sw1_height_ft etc) plus lead_hours and the
    upstream raw values for downstream comparison."""
    if not euro_recs or not gfs_recs:
        return []
    _warn_if_stale(Path(model_dir), cycle_utc)
    e = _records_to_frame(euro_recs, buoy_id, "EURO", cycle_utc)
    g = _records_to_frame(gfs_recs,  buoy_id, "GFS",  cycle_utc)
    if e.empty or g.empty:
        return []
    df = _join_models(e, g)
    if df.empty:
        return []
    df = _apply_dashboard_fallback_gfs(df)
    # The trainer's add_features needs every buoy_<id> column; the buoy
    # one-hot will be 1.0 for our buoy_id and 0.0 for the others.
    df = add_features(df)

    arch = detect_arch(Path(model_dir))
    if arch == "baseline":
        pred = predict_baseline(df, _load_baseline(Path(model_dir)))
    else:
        pred = predict_ml(df, _load_ml(Path(model_dir)))

    rows = []
    for i in df.index:
        rec = {
            "valid_utc":   df.at[i, "valid_utc"],
            "lead_hours":  int(df.at[i, "lead_hours"]) if not pd.isna(df.at[i, "lead_hours"]) else None,
            # Raw upstream side-by-side with the correction.
            "euro_sw1_height_ft":     _f(df.at[i, "euro_sw1_height_ft"]),
            "euro_sw1_period_s":      _f(df.at[i, "euro_sw1_period_s"]),
            "euro_sw1_direction_deg": _f(df.at[i, "euro_sw1_direction_deg"]),
            "gfs_sw1_height_ft":      _f(df.at[i, "gfs_sw1_height_ft"]),
            "gfs_sw1_period_s":       _f(df.at[i, "gfs_sw1_period_s"]),
            "gfs_sw1_direction_deg":  _f(df.at[i, "gfs_sw1_direction_deg"]),
            "gfs_sw1_source":         (df.at[i, "gfs_sw1_source"]
                                       if "gfs_sw1_source" in df.columns else "partition"),
        }
        for v in OUTPUT_DASHBOARD_VARS:
            col = f"pred_{v}"
            rec[f"csc2_{v}"] = _f(pred.at[i, col]) if col in pred.columns else None
        rows.append(rec)
    return rows


def _f(x):
    if pd.isna(x):
        return None
    return float(x)


_STALE_MODEL_DAYS = 120
_stale_warned: set[str] = set()


def _warn_if_stale(model_dir: Path, cycle_utc: str) -> None:
    """Log (once per model per process) when a model's training date is far
    behind the forecast cycle — a nudge that the drift watchdog / retrain
    trigger should have fired."""
    if model_dir.name in _stale_warned:
        return
    try:
        meta = json.loads((model_dir / "meta.json").read_text())
        trained = datetime.strptime(meta["trained_utc"], "%Y-%m-%dT%H:%M:%SZ")
        cycle = datetime.strptime(cycle_utc, "%Y%m%dT%HZ")
        age_days = (cycle - trained).days
    except Exception:
        return
    if age_days > _STALE_MODEL_DAYS:
        _stale_warned.add(model_dir.name)
        print(f"[csc2.predict] {model_dir.name} trained {age_days}d before "
              f"cycle {cycle_utc} — consider retraining")
