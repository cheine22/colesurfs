"""Multi-variant CSC serving — load every preserved model under
`.csc_models/current/` and produce per-variant live predictions.

Used by /api/csc/variants/<buoy_id> to let the dashboard compare multiple
trained CSC variants side-by-side on the live forecast.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cache import ttl_cache

from csc.display_filter import is_public_variant
from csc.features import add_engineered
from csc.models import load_model
from csc.predict import _primary_swell_lookup, _records_to_frame
from csc.schema import BUOYS, CSC_MODELS_DIR
from csc.serve import _strip_nans


FT_PER_M = 3.28084

_VARIANT_CACHE: dict[str, Any] = {"dir": None, "mtime": None, "models": {},
                                  "target": "combined"}


def _current_dir() -> Path | None:
    """Return the active artifact dir for the CSC /csc dashboard.

    Prefers `.csc_models/current_primary/` (v2, primary-swell target) when
    present so /csc displays primary-swell-trained predictions. Falls back
    to `.csc_models/current/` (v1, combined WVHT target) when v2 hasn't
    been promoted yet. This is the ONLY place where /csc switches target;
    `csc.serve` and `csc.predict` intentionally stay on v1 until the user
    flips them manually.
    """
    for name in ("current_primary", "current"):
        p = CSC_MODELS_DIR / name
        if p.exists():
            try:
                return p.resolve()
            except OSError:
                continue
    return None


def _active_target() -> str:
    """Read the currently-promoted artifact's `manifest.json::target` field
    to decide what quantity the dashboard's GFS/EURO traces should display.

    v3 onward: there is only one `current` symlink per scope, and its
    manifest declares `target='primary_swell'` or `target='combined'`.
    Pre-v3 fallback: also check for a legacy `current_primary` symlink.
    """
    import json
    cur = CSC_MODELS_DIR / "current"
    if cur.exists():
        try:
            mf = json.loads((cur / "manifest.json").read_text())
            if mf.get("target") == "primary_swell":
                return "primary_swell"
        except Exception:
            pass
    # Legacy fallback: pre-v3 used a parallel current_primary symlink.
    if (CSC_MODELS_DIR / "current_primary").exists():
        return "primary_swell"
    return "combined"


def _discover_variants() -> dict[str, Any]:
    """Return {variant_name: loaded_model}. Hot-reloads when the active
    symlink target changes. Each subdir with a `kind` file is treated as
    a variant; baselines (raw_gfs/raw_euro/mean/persistence and the v2
    partition-aware variants) are skipped because the dashboard renders
    those separately."""
    cur = _current_dir()
    if cur is None:
        return {}
    mtime = cur.stat().st_mtime
    target = _active_target()
    if (_VARIANT_CACHE.get("dir") == cur and
            _VARIANT_CACHE.get("mtime") == mtime and
            _VARIANT_CACHE.get("target") == target):
        return _VARIANT_CACHE["models"]
    models: dict[str, Any] = {}
    skip = {"raw_gfs", "raw_euro", "mean", "persistence",
            "raw_gfs_primary", "raw_euro_primary"}
    seen_targets: set[str] = set()   # dedupe backward-compat symlinks
    for sub in sorted(cur.iterdir()):
        if not sub.is_dir():
            continue
        # Skip symlinks whose target we've already loaded under its canonical
        # (dated) name — prevents the same model appearing twice once as e.g.
        # "funplus" and again as "csc_funplus_2026-04-20".
        if sub.is_symlink():
            try:
                tgt = str(sub.resolve())
                if tgt in seen_targets:
                    continue
            except OSError:
                continue
        kind_file = sub / "kind"
        if not kind_file.exists():
            continue
        kind = kind_file.read_text().strip()
        if kind in skip:
            continue
        if not is_public_variant(sub.name):
            continue
        try:
            models[sub.name] = load_model(sub)
            try:
                seen_targets.add(str(sub.resolve()))
            except OSError:
                pass
        except Exception:
            traceback.print_exc()
            continue
    _VARIANT_CACHE.update({"dir": cur, "mtime": mtime, "models": models,
                           "target": target})
    return models


def _region_name_for_buoy(buoy_id: str, spots: list[dict]) -> str | None:
    for s in spots:
        if str(s.get("buoy_id", "")) == buoy_id:
            return s.get("name")
    return None


_BUOY_BY_ID = {b[0]: b for b in BUOYS}


def _fetch_live_records(buoy_id: str) -> tuple[list[dict], list[dict]]:
    """Return (gfs_records, euro_records) from the warm waves.py cache,
    or direct Open-Meteo for CSC-only unregistered buoys. Mirrors the
    dispatch in csc.serve.fetch_csc_forecast without re-running the CSC
    correction."""
    from waves import fetch_all_wave_forecasts
    from config import SPOTS
    region = _region_name_for_buoy(buoy_id, SPOTS)
    if region is not None:
        gfs_by_spot = fetch_all_wave_forecasts("GFS") or {}
        euro_by_spot = fetch_all_wave_forecasts("EURO") or {}
        return gfs_by_spot.get(region) or [], euro_by_spot.get(region) or []
    # unregistered CSC-only buoy path — re-use the direct-OM fetcher
    from csc.serve import _fetch_csc_for_unregistered_buoy
    # We can't cheaply decouple the fetcher from the predict step there;
    # but the corrected output includes the raw gfs/euro values we need.
    # Rather than refactoring, hit Open-Meteo directly here.
    import requests
    b = _BUOY_BY_ID.get(buoy_id)
    if b is None:
        return [], []
    _, _, lat, lon, _ = b
    from csc.schema import OM_MODELS, OM_WAVE_VARS
    from config import FORECAST_DAYS

    def _fetch(model_key: str) -> list[dict]:
        try:
            r = requests.get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={
                    "latitude": lat, "longitude": lon,
                    "models": OM_MODELS[model_key],
                    "hourly": ",".join(OM_WAVE_VARS),
                    "forecast_days": FORECAST_DAYS,
                    "timezone": "UTC",
                },
                timeout=30,
            )
            r.raise_for_status()
            j = r.json()
        except Exception:
            return []
        h = j.get("hourly") or {}
        times = h.get("time") or []
        wh = h.get("wave_height") or [None] * len(times)
        wp = h.get("wave_period") or [None] * len(times)
        wd = h.get("wave_direction") or [None] * len(times)
        out = []
        for t, a, b_, c_ in zip(times, wh, wp, wd):
            out.append({"time": t, "wave_height_m": a,
                        "wave_period_s": b_, "wave_direction_deg": c_})
        return out
    return _fetch("GFS"), _fetch("EURO")


@ttl_cache(ttl_seconds=600, skip_none=True)
def fetch_all_variants_live(buoy_id: str) -> dict | None:
    """Produce a per-record payload with predictions from every CSC variant
    plus the raw GFS/EURO baselines and their 50/50 mean.

    Shape:
        {buoy_id, variants: ["ridge_mos","lgbm",...],
         records: [{valid_utc,
                    hs: {gfs, euro, mean, <variant>:..},
                    tp: {...}, dp: {...}}, ...]}
    """
    gfs_records, euro_records = _fetch_live_records(buoy_id)
    if not gfs_records or not euro_records:
        return None
    frame = _records_to_frame(buoy_id, gfs_records, euro_records)
    if frame.empty:
        return None
    eng = add_engineered(frame).reset_index(drop=True)

    # Target mode comes from which symlink is present:
    #   current_primary → primary-swell (v2) — GFS Hs sourced from the
    #                     `gfs_swell_wave_*` partition so all three traces
    #                     sit on the same surfable-component convention.
    #                     EURO has no swell partition via Open-Meteo — we
    #                     keep EURO's combined value and flag the caveat
    #                     via the top-level `"target"` field.
    #   current         → combined WVHT (v1, legacy) — every trace on
    #                     combined so GFS/EURO/CSC are apples-to-apples in
    #                     the same quantity, just a different one from the
    #                     main site's primary-swell Hs.
    target = _active_target()
    variants = _discover_variants()
    per_variant_preds: dict[str, pd.DataFrame] = {}
    for name, model in variants.items():
        try:
            per_variant_preds[name] = model.predict(eng).reset_index(drop=True)
        except Exception:
            traceback.print_exc()
            continue

    def _circ_mean(a, b):
        import numpy as np
        sa, ca = np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))
        sb, cb = np.sin(np.deg2rad(b)), np.cos(np.deg2rad(b))
        return (np.rad2deg(np.arctan2((sa + sb) / 2, (ca + cb) / 2)) + 360.0) % 360.0

    def _gfs_cell(row: "pd.Series", primary_col: str, combined_col: str):
        if target == "primary_swell" and primary_col in row.index:
            v = row.get(primary_col)
            if v is not None and pd.notna(v):
                return float(v)
        v = row.get(combined_col)
        return float(v) if v is not None and pd.notna(v) else None

    records: list[dict] = []
    for i, row in eng.iterrows():
        hs: dict[str, float | None] = {}
        tp: dict[str, float | None] = {}
        dp: dict[str, float | None] = {}
        gfs_hs_m = _gfs_cell(row, "gfs_swell_wave_height", "gfs_wave_height")
        gfs_tp_s = _gfs_cell(row, "gfs_swell_wave_period", "gfs_wave_period")
        gfs_dp_deg = _gfs_cell(row, "gfs_swell_wave_direction",
                               "gfs_wave_direction")
        gfs_hs_ft = gfs_hs_m * FT_PER_M if gfs_hs_m is not None else None
        euro_hs_ft = (float(row["euro_wave_height"]) * FT_PER_M
                      if pd.notna(row["euro_wave_height"]) else None)
        euro_tp_s = (float(row["euro_wave_period"])
                     if pd.notna(row["euro_wave_period"]) else None)
        euro_dp_deg = (float(row["euro_wave_direction"])
                       if pd.notna(row["euro_wave_direction"]) else None)
        hs["gfs"] = gfs_hs_ft
        hs["euro"] = euro_hs_ft
        hs["mean"] = (((hs["gfs"] or 0) + (hs["euro"] or 0)) / 2.0
                      if hs["gfs"] is not None and hs["euro"] is not None else None)
        tp["gfs"] = gfs_tp_s
        tp["euro"] = euro_tp_s
        tp["mean"] = ((tp["gfs"] + tp["euro"]) / 2.0
                      if tp["gfs"] is not None and tp["euro"] is not None else None)
        dp["gfs"] = gfs_dp_deg
        dp["euro"] = euro_dp_deg
        dp["mean"] = (float(_circ_mean(dp["gfs"], dp["euro"]))
                      if dp["gfs"] is not None and dp["euro"] is not None else None)
        for name, preds in per_variant_preds.items():
            hs[name] = float(preds.at[i, "pred_hs_m"]) * FT_PER_M
            tp[name] = float(preds.at[i, "pred_tp_s"])
            dp[name] = float(preds.at[i, "pred_dp_deg"])
        records.append({"valid_utc": row["valid_utc"].isoformat(),
                        "hs": hs, "tp": tp, "dp": dp})

    payload = {
        "buoy_id": buoy_id,
        "variants": list(per_variant_preds.keys()),
        "target": target,
        "records": records,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return _strip_nans(payload)
