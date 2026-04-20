"""CSC inference — take live GFS and Euro forecast records (one hourly
record per valid_utc, `waves.py` output shape) and emit a corrected
forecast by applying the currently-promoted CSC model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csc.features import add_engineered
from csc.models import load_model
from csc.schema import BUOYS, CSC_MODELS_DIR


# ─── Artifact cache (hot-reload on symlink mtime change) ──────────────────

_ARTIFACT_CACHE: dict[str, Any] = {"path": None, "mtime": None, "model": None,
                                   "manifest": None}


def _current_artifact_path() -> Path | None:
    p = CSC_MODELS_DIR / "current"
    if not p.exists():
        return None
    try:
        resolved = p.resolve()
    except OSError:
        return None
    return resolved


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


# ─── Live-forecast → wide feature frame ───────────────────────────────────

def _rename_model_frame(records: list[dict], prefix: str) -> pd.DataFrame:
    """Project a live-forecast record list onto the column schema the
    trained CSC model expects.

    The v1 training frame used Open-Meteo's raw hourly keys (wave_height,
    wave_period, wave_direction) which are the COMBINED sea-state values.
    The v2 training frame additionally uses swell_wave_* partitions when
    available (GFS only — ECMWF WAM has no partitions). Waves.py records
    carry:
      - `combined_wave_height_m` etc. (combined wave field, m)
      - `wave_height_ft` etc.  (primary-swell partition, ft, filtered >6s)
      - `components: [{height_ft, period_s, direction_deg}, ...]`
    We map both combined AND primary-swell onto the training schema so
    multi_serve.py can display whichever quantity the currently-active
    CSC target expects.
    """
    df = pd.DataFrame(records)
    # Combined (v1 training target)
    rename_combined = {
        "combined_wave_height_m":   f"{prefix}_wave_height",
        "combined_wave_period_s":   f"{prefix}_wave_period",
        "combined_wave_direction_deg": f"{prefix}_wave_direction",
    }
    rename_raw = {
        "wave_height_m":   f"{prefix}_wave_height",
        "wave_period_s":   f"{prefix}_wave_period",
        "wave_direction_deg": f"{prefix}_wave_direction",
    }
    for src, dst in rename_combined.items():
        if src in df.columns:
            df[dst] = df[src]
    for src, dst in rename_raw.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    # Primary-swell partition (v2 training target). waves.py stores it as
    # `wave_height_ft` at top-level (pulled from components[0]) — convert
    # back to meters so the column aligns with the training frame.
    if "wave_height_ft" in df.columns:
        df[f"{prefix}_swell_wave_height"] = df["wave_height_ft"] / 3.28084
    if "wave_period_s" in df.columns:
        df[f"{prefix}_swell_wave_period"] = df["wave_period_s"]
    if "wave_direction_deg" in df.columns:
        df[f"{prefix}_swell_wave_direction"] = df["wave_direction_deg"]
    if "time" in df.columns and "valid_utc" not in df.columns:
        df.rename(columns={"time": "valid_utc"}, inplace=True)
    return df


def _primary_swell_lookup(records: list[dict]) -> dict[pd.Timestamp, dict]:
    """Index a waves.py record list by timestamp → primary-swell Hs/Tp/Dp.

    The main dashboard renders `wave_height_ft` / `wave_period_s` /
    `wave_direction_deg` from each record (populated by waves.py from the
    highest-energy swell partition). We look those up here so the CSC live
    forecast shows GFS/EURO traces using the same surfable-component
    convention. Records from the unregistered-buoy path (direct Open-Meteo)
    lack these keys — the caller falls back to combined values in that case.

    Keys are UTC-normalized pandas Timestamps so they match the merged frame's
    `valid_utc` column exactly (waves.py uses bare ISO strings like
    ``2026-04-20T00:00``; the merged frame is tz-aware UTC).
    """
    out: dict[pd.Timestamp, dict] = {}
    for r in records:
        t = r.get("time") or r.get("valid_utc")
        if t is None:
            continue
        try:
            ts = pd.to_datetime(t, utc=True)
        except (ValueError, TypeError):
            continue
        out[ts] = {
            "hs_ft":  r.get("wave_height_ft"),
            "tp_s":   r.get("wave_period_s"),
            "dp_deg": r.get("wave_direction_deg"),
        }
    return out


def _records_to_frame(buoy_id: str,
                      gfs_records: list[dict],
                      euro_records: list[dict]) -> pd.DataFrame:
    """Pair hourly GFS + Euro records by timestamp into the wide schema
    csc.features.add_engineered expects."""
    g = _rename_model_frame(gfs_records, "gfs")
    e = _rename_model_frame(euro_records, "euro")

    # GFS has swell partitions when fetched via the CSC-only path (or any
    # direct OM call that requested them). Fold those into gfs_* cols.
    swell_map = {
        "swell_wave_height":       "gfs_swell_wave_height",
        "swell_wave_period":       "gfs_swell_wave_period",
        "swell_wave_direction":    "gfs_swell_wave_direction",
        "secondary_swell_wave_height":    "gfs_secondary_swell_wave_height",
        "secondary_swell_wave_period":    "gfs_secondary_swell_wave_period",
        "secondary_swell_wave_direction": "gfs_secondary_swell_wave_direction",
    }
    for src, dst in swell_map.items():
        if src in g.columns and dst not in g.columns:
            g[dst] = g[src]

    g = g[[c for c in g.columns if c.startswith("gfs_") or c == "valid_utc"]]
    e = e[[c for c in e.columns if c.startswith("euro_") or c == "valid_utc"]]
    merged = pd.merge(g, e, on="valid_utc", how="inner")
    merged["valid_utc"] = pd.to_datetime(merged["valid_utc"], utc=True)
    merged["buoy_id"] = buoy_id
    # fill obs_* with NaN so add_engineered's obs_sin/cos_dp don't blow up
    merged["obs_hs_m"] = np.nan
    merged["obs_tp_s"] = np.nan
    merged["obs_dp_deg"] = np.nan
    return merged


def correct_forecast(buoy_id: str,
                     gfs_records: list[dict],
                     euro_records: list[dict]) -> dict[str, Any]:
    """Produce a corrected forecast record list mirroring waves.py shape."""
    _maybe_reload()
    model = _ARTIFACT_CACHE.get("model")
    manifest = _ARTIFACT_CACHE.get("manifest")
    if model is None or not gfs_records or not euro_records:
        return {
            "buoy_id": buoy_id,
            "records": [],
            "artifact_version": None,
            "fallback": "no_model_or_records",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    frame = _records_to_frame(buoy_id, gfs_records, euro_records)
    if frame.empty:
        return {"buoy_id": buoy_id, "records": [],
                "artifact_version": manifest.get("version") if manifest else None,
                "fallback": "no_aligned_records",
                "generated_at": datetime.now(timezone.utc).isoformat()}

    eng = add_engineered(frame)
    preds = model.predict(eng)
    eng = eng.reset_index(drop=True)
    preds = preds.reset_index(drop=True)

    # Display layer: CSC is trained on NDBC combined WVHT, so every trace on
    # /csc reports combined Hs (total sea state) for apples-to-apples
    # comparison across GFS / EURO / CSC. The main dashboard shows primary-
    # swell Hs (surfable partition) which is a different quantity; the
    # footer on /csc explains the gap. Mixing primary and combined on the
    # same chart makes EURO look 3× bigger than GFS because Open-Meteo's
    # ECMWF WAM does not expose swell partitions — avoid that trap.
    out_records = []
    for i, row in eng.iterrows():
        ts_key = row["valid_utc"]
        gfs_hs = (float(row["gfs_wave_height"]) * 3.28084
                  if pd.notna(row["gfs_wave_height"]) else None)
        euro_hs = (float(row["euro_wave_height"]) * 3.28084
                   if pd.notna(row["euro_wave_height"]) else None)
        gfs_tp = (float(row["gfs_wave_period"])
                  if pd.notna(row["gfs_wave_period"]) else None)
        euro_tp = (float(row["euro_wave_period"])
                   if pd.notna(row["euro_wave_period"]) else None)
        gfs_dp = (float(row["gfs_wave_direction"])
                  if pd.notna(row["gfs_wave_direction"]) else None)
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
        "fallback": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
