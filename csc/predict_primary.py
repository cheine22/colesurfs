"""CSC v2 inference — primary-swell-target corrected forecast.

Mirrors `csc.predict` one-for-one but:
  * loads the artifact from `.csc_models/current_primary/` (distinct from
    v1's `current` so the two can co-exist until v2 is explicitly promoted)
  * tags the emitted records with `"target": "primary_swell"` for the
    dashboard to label.

This module intentionally imports helpers from `csc.predict`
(`_records_to_frame`, `_primary_swell_lookup`) rather than duplicating the
live-forecast → feature-frame projection, so any future schema tweak lands
in one place.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from csc.features import add_engineered
from csc.models import load_model
from csc.predict import _records_to_frame
from csc.schema import CSC_MODELS_DIR


_ARTIFACT_CACHE: dict[str, Any] = {"path": None, "mtime": None, "model": None,
                                   "manifest": None}


def _current_artifact_path() -> Path | None:
    p = CSC_MODELS_DIR / "current_primary"
    if not p.exists():
        return None
    try:
        return p.resolve()
    except OSError:
        return None


def _maybe_reload() -> None:
    path = _current_artifact_path()
    if path is None:
        _ARTIFACT_CACHE.update({"path": None, "mtime": None, "model": None,
                                "manifest": None})
        return
    mtime = path.stat().st_mtime
    if _ARTIFACT_CACHE.get("path") == path and _ARTIFACT_CACHE.get("mtime") == mtime:
        return
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        _ARTIFACT_CACHE.update({"path": None, "mtime": None, "model": None,
                                "manifest": None})
        return
    manifest = json.loads(manifest_path.read_text())
    winner_dir = Path(manifest["winner_dir"])
    if not winner_dir.is_absolute():
        winner_dir = path / manifest["winner"]
    model = load_model(winner_dir)
    _ARTIFACT_CACHE.update({
        "path": path, "mtime": mtime, "model": model, "manifest": manifest,
    })


def current_manifest() -> dict[str, Any] | None:
    _maybe_reload()
    return _ARTIFACT_CACHE.get("manifest")


def has_v2_artifact() -> bool:
    """True iff `.csc_models/current_primary/` exists and loads cleanly."""
    _maybe_reload()
    return _ARTIFACT_CACHE.get("model") is not None


def correct_forecast_primary(buoy_id: str,
                             gfs_records: list[dict],
                             euro_records: list[dict]) -> dict[str, Any]:
    """Produce a primary-swell-targeted corrected forecast.

    The record shape mirrors `csc.predict.correct_forecast` — `hs/tp/dp`
    dicts keyed by `csc`/`gfs`/`euro` — plus a top-level `"target":
    "primary_swell"` so the dashboard can label these traces without
    inferring from the model path.

    GFS `hs` is sourced from the primary-swell partition
    (`swell_wave_height`) when available, falling back to combined
    `wave_height`. EURO has no swell partition exposed by Open-Meteo, so
    EURO `hs` continues to report combined `wave_height`; the dashboard
    caller should label this caveat in the UI.
    """
    _maybe_reload()
    model = _ARTIFACT_CACHE.get("model")
    manifest = _ARTIFACT_CACHE.get("manifest")
    if model is None or not gfs_records or not euro_records:
        return {
            "buoy_id": buoy_id,
            "records": [],
            "artifact_version": None,
            "target": "primary_swell",
            "fallback": "no_model_or_records",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    frame = _records_to_frame(buoy_id, gfs_records, euro_records)
    if frame.empty:
        return {"buoy_id": buoy_id, "records": [],
                "artifact_version": manifest.get("version") if manifest else None,
                "target": "primary_swell",
                "fallback": "no_aligned_records",
                "generated_at": datetime.now(timezone.utc).isoformat()}

    eng = add_engineered(frame)
    preds = model.predict(eng)
    eng = eng.reset_index(drop=True)
    preds = preds.reset_index(drop=True)

    out_records: list[dict[str, Any]] = []
    for i, row in eng.iterrows():
        ts_key = row["valid_utc"]

        # GFS: prefer primary-swell partition (matches /csc training target
        # and main-dashboard primary-swell Hs display).
        def _g(col_primary: str, col_combined: str):
            v = row.get(col_primary)
            if v is None or pd.isna(v):
                v = row.get(col_combined)
            return v

        gfs_hs_m = _g("gfs_swell_wave_height", "gfs_wave_height")
        gfs_tp_s = _g("gfs_swell_wave_period", "gfs_wave_period")
        gfs_dp = _g("gfs_swell_wave_direction", "gfs_wave_direction")
        gfs_hs = (float(gfs_hs_m) * 3.28084
                  if gfs_hs_m is not None and pd.notna(gfs_hs_m) else None)
        gfs_tp = float(gfs_tp_s) if gfs_tp_s is not None and pd.notna(gfs_tp_s) else None
        gfs_dp = float(gfs_dp) if gfs_dp is not None and pd.notna(gfs_dp) else None

        euro_hs = (float(row["euro_wave_height"]) * 3.28084
                   if pd.notna(row["euro_wave_height"]) else None)
        euro_tp = (float(row["euro_wave_period"])
                   if pd.notna(row["euro_wave_period"]) else None)
        euro_dp = (float(row["euro_wave_direction"])
                   if pd.notna(row["euro_wave_direction"]) else None)

        out_records.append({
            "valid_utc": ts_key.isoformat(),
            "hs": {
                "csc":  float(preds.at[i, "pred_hs_m"]) * 3.28084,
                "gfs":  gfs_hs,
                "euro": euro_hs,
            },
            "tp": {
                "csc":  float(preds.at[i, "pred_tp_s"]),
                "gfs":  gfs_tp,
                "euro": euro_tp,
            },
            "dp": {
                "csc":  float(preds.at[i, "pred_dp_deg"]),
                "gfs":  gfs_dp,
                "euro": euro_dp,
            },
        })

    return {
        "buoy_id": buoy_id,
        "records": out_records,
        "artifact_version": manifest.get("version") if manifest else None,
        "target": "primary_swell",
        "fallback": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
