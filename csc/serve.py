"""CSC serving glue — TTL-cached Flask-facing fetcher.

Consumes waves.py's per-region cache and applies the currently-promoted
CSC artifact via csc.predict.correct_forecast. No extra network I/O —
reads live forecasts from the main app's in-memory cache.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _strip_nans(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None so the result is valid JSON.
    Python's json module emits NaN tokens by default and Flask's jsonify
    inherits that behavior — the browser's JSON.parse rejects them."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _strip_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_nans(v) for v in obj]
    return obj

from cache import ttl_cache

from csc.predict import correct_forecast, current_manifest
from csc.schema import BUOYS, CSC_DATA_DIR, CSC_MODELS_DIR


_BUOY_BY_ID = {b[0]: b for b in BUOYS}


def _region_name_for_buoy(buoy_id: str, spots: list[dict]) -> str | None:
    """Match a CSC buoy_id to a waves.py 'spot name' via SPOTS config."""
    for s in spots:
        if str(s.get("buoy_id", "")) == buoy_id:
            return s.get("name")
    return None


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_csc_forecast(buoy_id: str) -> dict | None:
    """Main inference entry point. Returns dict with shape:
      {buoy_id, records:[{valid_utc, hs:{csc,gfs,euro}, tp:{...}, dp:{...}}],
       artifact_version, fallback, generated_at}
    or None if either model's live cache is empty.
    """
    from waves import fetch_all_wave_forecasts
    from config import SPOTS
    region = _region_name_for_buoy(buoy_id, SPOTS)
    if region is None:
        # CSC-only buoys (46221, 46268 — added to CSC but not regions.yaml)
        return _fetch_csc_for_unregistered_buoy(buoy_id)

    gfs_by_spot = fetch_all_wave_forecasts("GFS") or {}
    euro_by_spot = fetch_all_wave_forecasts("EURO") or {}
    gfs = gfs_by_spot.get(region) or []
    euro = euro_by_spot.get(region) or []
    return correct_forecast(buoy_id, gfs, euro)


@ttl_cache(ttl_seconds=1800, skip_none=True)
def _fetch_csc_for_unregistered_buoy(buoy_id: str) -> dict | None:
    """CSC-scope-only buoys (not on the main dashboard): call the Open-Meteo
    Marine API directly for both models and apply the correction."""
    import requests
    b = _BUOY_BY_ID.get(buoy_id)
    if b is None:
        return None
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
            out.append({
                "time": t,
                # Keys below match csc.predict._records_to_frame's rename map
                # so the feature-frame has populated gfs_*/euro_* columns.
                "wave_height_m":   a,
                "wave_period_s":   b_,
                "wave_direction_deg": c_,
            })
        return out

    gfs = _fetch("GFS")
    euro = _fetch("EURO")
    return correct_forecast(buoy_id, gfs, euro)


# ─── Comparison dashboard payload ─────────────────────────────────────────

def list_trained_versions() -> list[dict[str, Any]]:
    """Return metadata for every preserved model artifact under .csc_models/
    (sorted newest first). Each entry has `version`, `winner`, `buoys`, and
    the full metrics bundle — the comparison dashboard reads this to
    populate the stats table row-by-row."""
    root = Path(CSC_MODELS_DIR)
    if not root.exists():
        return []
    versions: list[dict[str, Any]] = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name == "current":
            continue
        mf = d / "manifest.json"
        mt = d / "metrics.json"
        if not mf.exists() or not mt.exists():
            continue
        manifest = json.loads(mf.read_text())
        metrics = json.loads(mt.read_text())
        versions.append({
            "version": manifest.get("version") or d.name,
            "winner": manifest.get("winner"),
            "dir": d.name,
            "scope": manifest.get("scope", "all"),
            "target": manifest.get("target") or metrics.get("target", "combined"),
            "buoys": manifest.get("buoys", []),
            "generated_at": manifest.get("generated_at"),
            "train_rows": manifest.get("train_rows"),
            "test_rows": manifest.get("test_rows"),
            "holdout_cutoff": manifest.get("holdout_cutoff"),
            "metrics": metrics.get("metrics", {}),
        })
    return versions


def compare_payload(buoy_id: str) -> dict[str, Any]:
    """Payload backing /api/csc/compare?buoy_id=... — live forecast + every
    trained version's metrics bundle + current-artifact tag."""
    live = fetch_csc_forecast(buoy_id) or {"records": [], "fallback": "no_live_data"}
    versions = list_trained_versions()
    manifest = current_manifest()
    return _strip_nans({
        "buoy_id": buoy_id,
        "live": live,
        "versions": versions,
        "current_version": (manifest or {}).get("version"),
        "current_winner": (manifest or {}).get("winner"),
    })
