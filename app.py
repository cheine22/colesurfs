"""
colesurfs — Flask Application

Routes:
  /                              Dashboard UI (single-page app)
  /api/buoys                     Live NOAA buoy readings for all regions
  /api/forecast/<EURO|GFS>       10-day hourly wave forecast per buoy
  /api/wind?model=               Current wind snapshot for map init
  /api/wind_forecast?model=      Full hourly wind grid (for hover-sync)
  /api/wind_spots                Hourly wind forecast per buoy location
  /api/region_wind?model=        Hourly wind per surf spot (regional mode)
  /api/tides                     Per-spot tide predictions with Surfline corrections
  /api/config                    Spots, swell categories, wind bands, region views
  /api/sun                       Sunrise/sunset (computed locally, no external API)
  /api/status?model=             Model run estimate + daily API usage
  /api/debug/spectral/<id>       Diagnostic: raw spectral parse (COLESURFS_DEBUG=1 only)
  /api/buoy_history/<station_id>  10-day historical buoy data with spectral components (?days= override)
  /api/buoy_historical_context    Historical obs + per-hour model_agreement vs CSC2 archives
  /api/refresh (POST)            Clear caches + reload swell rules

v1.5: EURO wave forecast migrated from Open-Meteo ECMWF-WAM to Copernicus Marine
      (CMEMS) ECMWF-WAM ANFC. Open-Meteo EURO waves were removed from the site
      entirely — OM returns null swell partitions, so switching to CMEMS gives
      us real SW1/SW2 spectral partitions. GFS continues via Open-Meteo.
"""
import json as _json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_compress import Compress
from waitress import serve

from config import SPOTS, WIND_SPOTS, MODEL_COLORS, WIND_BANDS, REGION_VIEWS
import swell_rules
from buoy  import (fetch_buoy, fetch_buoy_history, fetch_buoy_historical_context,
                    _parse_spectral_file, _spectral_components)
import requests as _req_buoy
from waves import fetch_wave_forecast, fetch_all_wave_forecasts
from waves_cmems import fetch_all_cmems_wave_forecasts
from wind  import (fetch_wind_grid, fetch_spot_wind,
                    fetch_wind_forecast_grid, fetch_spot_wind_forecasts,
                    fetch_region_wind_forecasts, estimate_model_run)
from tide  import fetch_tide_predictions
from sun   import compute_sun_data
import cache as _cache

app = Flask(__name__)
Compress(app)   # gzip/brotli compression for all responses > 500 bytes
PORT = int(os.environ.get("COLESURFS_PORT", 5151))
# Bind to 0.0.0.0 so LAN devices can reach /tuner at http://<mac-ip>:5151.
# Public internet access keeps going through the Cloudflare tunnel →
# loopback, which we gate /tuner against via _restrict_tuner below.
HOST = os.environ.get("COLESURFS_HOST", "0.0.0.0")
DEBUG_MODE = os.environ.get("COLESURFS_DEBUG", "").strip() == "1"

# Thread pool for parallel buoy fetches (reused across requests)
_buoy_pool = ThreadPoolExecutor(max_workers=8)


# ─── Rate limiter ────────────────────────────────────────────────────────────
_rate_lock = threading.Lock()
_rate_hits: dict[str, list[float]] = {}   # "bucket:ip" → [timestamps]


def _rate_limited(bucket: str, ip: str, max_calls: int, window_sec: int) -> bool:
    """Return True if this IP has exceeded max_calls in the last window_sec seconds."""
    key = f"{bucket}:{ip}"
    now = time.monotonic()
    with _rate_lock:
        hits = _rate_hits.get(key, [])
        hits = [t for t in hits if now - t < window_sec]
        if len(hits) >= max_calls:
            _rate_hits[key] = hits
            return True
        hits.append(now)
        _rate_hits[key] = hits
        return False


@app.before_request
def _check_api_rate_limit():
    """General rate limit: 60 requests/minute per IP on /api/* routes."""
    if request.path.startswith("/api/"):
        ip = request.remote_addr or "unknown"
        if _rate_limited("api", ip, max_calls=60, window_sec=60):
            return jsonify({"error": "rate limit exceeded"}), 429


import ipaddress as _ipaddress


@app.before_request
def _restrict_tuner():
    """Gate /tuner + /api/tuner/* to LAN clients only. The Cloudflare tunnel
    proxies public requests through loopback, so we can't rely on remote_addr
    alone — we additionally reject any request carrying Cloudflare-set
    headers (CF-Ray, CF-Connecting-IP) or one where the Host header isn't
    a LAN/loopback address."""
    path = request.path or ""
    if path != "/tuner" and not path.startswith("/api/tuner/"):
        return None
    if request.headers.get("CF-Ray") or request.headers.get("CF-Connecting-IP"):
        return jsonify({"error": "not found"}), 404
    host = (request.host or "").split(":")[0].strip().lower()
    if host in ("localhost",):
        return None
    try:
        ip = _ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private:
            return None
    except ValueError:
        pass
    return jsonify({"error": "not found"}), 404


@app.route("/api/buoys")
def api_buoys():
    """Live NOAA buoy readings — fetched in parallel for speed."""
    futures = {
        _buoy_pool.submit(fetch_buoy, s["buoy_id"]): s["name"]
        for s in SPOTS
    }
    result = {}
    for future in as_completed(futures):
        name = futures[future]
        try:
            result[name] = future.result(timeout=20)
        except Exception as e:
            print(f"[buoys] {name} parallel fetch error: {e}")
            result[name] = None
    return jsonify(result)


@app.route("/api/forecast/<model_key>")
def api_forecast(model_key: str):
    """Model keys:
      EURO — Copernicus Marine ECMWF WAM ANFC (per-buoy 3h, interpolated hourly,
             SW1/SW2 partitions with Tm01→Tp scaled periods)
      GFS  — Open-Meteo NCEP GFS-Wave (per-spot hourly)

    v1.5: Open-Meteo ECMWF-WAM was removed from the site; EURO is now CMEMS
    exclusively. /api/forecast/C-EURO remains as a backward-compat alias.
    """
    key = model_key.upper()
    if key in ("EURO", "C-EURO"):
        return jsonify(fetch_all_cmems_wave_forecasts() or {})
    if key == "GFS":
        return jsonify(fetch_all_wave_forecasts("GFS") or {})
    return jsonify({"error": "unknown model"}), 400


@app.route("/api/wind")
def api_wind():
    model_key = request.args.get("model", "EURO").upper()
    if model_key not in ("EURO", "GFS"):
        model_key = "EURO"
    # Fetch spot winds in parallel to avoid sequential 429 waits
    spot_futures = {
        _buoy_pool.submit(fetch_spot_wind, s["lat"], s["lon"]): s["name"]
        for s in SPOTS
    }
    spot_winds = {}
    for future in as_completed(spot_futures, timeout=15):
        name = spot_futures[future]
        try:
            spot_winds[name] = future.result(timeout=12)
        except Exception:
            spot_winds[name] = None
    return jsonify({
        "grid":       fetch_wind_grid(model_key),
        "spot_winds": spot_winds,
    })


@app.route("/api/wind_forecast")
def api_wind_forecast():
    """Full hourly wind grid for hover-sync with swell table."""
    model_key = request.args.get("model", "EURO").upper()
    if model_key not in ("EURO", "GFS"):
        model_key = "EURO"
    return jsonify(fetch_wind_forecast_grid(model_key))


@app.route("/api/wind_spots")
def api_wind_spots():
    """Hourly wind forecast per configured spot — for the WIND table row."""
    return jsonify(fetch_spot_wind_forecasts())


@app.route("/api/region_wind")
def api_region_wind():
    """Hourly wind + gust forecast for all WIND_SPOTS — for Regional Mode table.

    Optional `past_days` (0..30) extends the response backwards so the
    dashboard's historical-data toggle can populate the -240h wind strip
    in Regional Mode using the same model's analysis hours.
    """
    model_key = request.args.get("model", "EURO").upper()
    if model_key not in ("EURO", "GFS"):
        model_key = "EURO"
    try:
        past_days = int(request.args.get("past_days", 0))
    except (TypeError, ValueError):
        past_days = 0
    past_days = max(0, min(past_days, 30))
    return jsonify(fetch_region_wind_forecasts(model_key, past_days=past_days))


@app.route("/api/tides")
def api_tides():
    """Hourly tide predictions (height ft + daily %) for all WIND_SPOT tide stations.

    Optional `past_days` (0..30) extends the begin_date backwards for the
    historical-data toggle.
    """
    try:
        past_days = int(request.args.get("past_days", 0))
    except (TypeError, ValueError):
        past_days = 0
    past_days = max(0, min(past_days, 30))
    return jsonify(fetch_tide_predictions(past_days=past_days))


@app.route("/api/status")
def api_status():
    """Model run estimate + API usage counter for the UI status bar."""
    model_key = request.args.get("model", "EURO").upper()
    if model_key not in ("EURO", "GFS"):
        model_key = "EURO"
    return jsonify({
        "model_run": estimate_model_run(model_key),
        "api_usage": _cache.get_api_usage(),
    })


@app.route("/api/sun")
def api_sun():
    """Sunrise/sunset for the forecast period, computed locally via astral."""
    # Use first spot's coordinates — all East Coast spots are close enough
    # that sunrise/sunset times differ by at most ~5 minutes.
    spot = SPOTS[0] if SPOTS else {"lat": 40.58, "lon": -73.63}
    return jsonify(compute_sun_data(spot["lat"], spot["lon"]))


@app.route("/api/config")
def api_config():
    import wind_rules  # lazy: keeps import cost off cold startup
    bands = swell_rules.load_bands()
    return jsonify({
        "spots": SPOTS,
        "swell_categories": [
            {
                "name":       cat,
                "dark_bg":    swell_rules.COLORS[cat]["dark_bg"],
                "dark_text":  swell_rules.COLORS[cat]["dark_text"],
                "light_bg":   swell_rules.COLORS[cat]["light_bg"],
                "light_text": swell_rules.COLORS[cat]["light_text"],
            }
            for cat in swell_rules.CATEGORIES
        ],
        "swell_bands": [
            {"period_ub": b["period_ub"], "rules": b["rules"]}
            for b in bands
        ],
        "wind_bands": [
            {"min": b[0], "max": b[1], "bg": b[2], "text": b[3]}
            for b in WIND_BANDS
        ],
        "wind_rating": wind_rules.load_config(),
        "model_colors": MODEL_COLORS,
        "wind_spots":   WIND_SPOTS,
        "region_views": REGION_VIEWS,
    })


@app.route("/api/debug/spectral/<station_id>")
def api_debug_spectral(station_id: str):
    """Diagnostic: fetch raw spectral files for a buoy and return parsed results.
    Only available when COLESURFS_DEBUG=1."""
    if not DEBUG_MODE:
        return jsonify({"error": "not found"}), 404
    hdrs = {"User-Agent": "ColeSurfs/1.0"}
    ds_url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.data_spec"
    sw_url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir"
    try:
        rds = _req_buoy.get(ds_url, timeout=15, headers=hdrs)
        rsw = _req_buoy.get(sw_url, timeout=15, headers=hdrs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    spec_bins  = _parse_spectral_file(rds.text, 1) if rds.status_code == 200 else []
    swdir_bins = _parse_spectral_file(rsw.text, 0) if rsw.status_code == 200 else []
    comps      = _spectral_components(spec_bins, swdir_bins) if spec_bins and swdir_bins else []

    swell_bins = [(f, e, d) for (f, e), (_, d) in zip(spec_bins, swdir_bins)
                  if f <= 1/6 + 0.015] if spec_bins and swdir_bins else []

    return jsonify({
        "station":        station_id,
        "data_spec_http": rds.status_code,
        "swdir_http":     rsw.status_code,
        "spec_bins":      len(spec_bins),
        "swdir_bins":     len(swdir_bins),
        "swell_band_bins": [
            {"freq": round(f, 4), "period": round(1/f, 1), "energy": e, "dir": d}
            for f, e, d in swell_bins if f > 0
        ],
        "components":     comps,
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Clear all caches and reload swell + wind rules. Rate-limited to 1 call per 30s."""
    ip = request.remote_addr or "unknown"
    if _rate_limited("refresh", ip, max_calls=1, window_sec=30):
        return jsonify({"error": "rate limit exceeded, try again in 30s"}), 429
    _cache.clear_all()
    swell_rules.reload()
    import wind_rules
    wind_rules.reload()
    return jsonify({"status": "cache cleared, swell + wind rules reloaded"})


@app.route("/api/buoy_history/<station_id>")
def api_buoy_history(station_id):
    """Historical buoy data with spectral swell components (default 10 days).
    Optional ?days= query arg lets callers tune the window."""
    valid_ids = {s["buoy_id"] for s in SPOTS}
    if station_id not in valid_ids:
        return jsonify({"error": "unknown station"}), 404
    try:
        days = int(request.args.get("days", 10))
    except (TypeError, ValueError):
        days = 10
    days = max(1, min(days, 45))   # NDBC realtime2 covers ~45 days
    data = fetch_buoy_history(station_id, days=days)
    if data is None:
        return jsonify({"error": "data unavailable"}), 503
    return jsonify(data)


@app.route("/api/buoy_historical_context")
def api_buoy_historical_context():
    """Observed history + per-record model_agreement vs local CSC2 archives.
    Only CSC2-scope buoys get non-null agreement; others return null."""
    station_id = request.args.get("station_id", "")
    valid_ids = {s["buoy_id"] for s in SPOTS}
    if station_id not in valid_ids:
        return jsonify({"error": "unknown station"}), 404
    try:
        days = int(request.args.get("days", 10))
    except (TypeError, ValueError):
        days = 10
    days = max(1, min(days, 45))
    data = fetch_buoy_historical_context(station_id, days=days)
    if data is None:
        return jsonify({"error": "data unavailable"}), 503
    return jsonify(data)


@app.route("/")
def index():
    # Inline /api/config data into the HTML to save one round-trip on initial load.
    import wind_rules
    bands = swell_rules.load_bands()
    config_data = {
        "spots": SPOTS,
        "swell_categories": [
            {
                "name":       cat,
                "dark_bg":    swell_rules.COLORS[cat]["dark_bg"],
                "dark_text":  swell_rules.COLORS[cat]["dark_text"],
                "light_bg":   swell_rules.COLORS[cat]["light_bg"],
                "light_text": swell_rules.COLORS[cat]["light_text"],
            }
            for cat in swell_rules.CATEGORIES
        ],
        "swell_bands": [
            {"period_ub": b["period_ub"], "rules": b["rules"]}
            for b in bands
        ],
        "wind_bands": [
            {"min": b[0], "max": b[1], "bg": b[2], "text": b[3]}
            for b in WIND_BANDS
        ],
        "wind_rating": wind_rules.load_config(),
        "model_colors": MODEL_COLORS,
        "wind_spots":   WIND_SPOTS,
        "region_views": REGION_VIEWS,
    }
    return render_template("index.html",
                           inline_config=_json.dumps(config_data, separators=(',', ':')))

@app.route("/csc")
def csc_page():
    """CSC2 evaluation page. Under construction — data collection in progress."""
    from csc2.schema import BUOYS as _CSC2_BUOYS
    buoys = [
        {"buoy_id": b[0], "label": b[1], "lat": b[2], "lon": b[3], "scope": b[4]}
        for b in _CSC2_BUOYS
    ]
    return render_template(
        "csc.html",
        inline_config=_json.dumps({"buoys": buoys}, separators=(',', ':')),
    )


@app.route("/csc-model")
def csc_model_page():
    """CSC2 model documentation — explains the training pipeline end-to-end."""
    return render_template("csc-model.html")


@app.route("/api/csc2/archive_status")
def api_csc2_archive_status():
    """How much CMEMS/GFS/buoy archive we've accumulated so far, per buoy."""
    from csc2.archive_status import summarize
    return jsonify(summarize())


@app.route("/api/csc2/models")
def api_csc2_models():
    """Trained-model registry. Returns the 3-model selection that surfaces
    on /csc — #1 by composite skill (vs raw EURO holdout MAE) plus the
    two most recent additional models — alongside the full inventory."""
    from csc2.registry import selection_payload
    scope = request.args.get("scope", "east")
    if scope not in ("east", "west"):
        scope = "east"
    return jsonify(selection_payload(scope))


@_cache.ttl_cache(ttl_seconds=1800, skip_none=True)
def _csc2_forecast_payload(buoy_id: str, scope: str) -> dict | None:
    """Compute /api/csc2/forecast and cache for 30 min.

    The cycle anchor only changes every 12 h (CMEMS publish cadence) so
    response-level caching is safe. The cache warmer pre-fills this for
    every east buoy on startup, so users essentially never pay the cold
    cost (~7 s for CMEMS + ~600 ms for the 3 model predictions)."""
    from csc2.registry import selection_payload
    from csc2.predict import predict_for_cycle
    from csc2.schema import buoy_meta, CSC2_MODELS_DIR
    from datetime import datetime, timezone as _tz
    from waves_cmems import fetch_cmems_point
    from waves import fetch_wave_forecast

    try:
        meta = buoy_meta(buoy_id)
    except KeyError:
        return None

    now = datetime.now(_tz.utc)
    cyc_h = 0 if now.hour < 12 else 12
    cycle_utc = now.replace(hour=cyc_h, minute=0, second=0, microsecond=0
                            ).strftime("%Y%m%dT%HZ")

    try:
        euro_recs = fetch_cmems_point(meta["lat"], meta["lon"]) or []
        euro_err = None
    except Exception as e:
        euro_recs, euro_err = [], f"{type(e).__name__}: {e}"
    try:
        gfs_recs = fetch_wave_forecast(meta["lat"], meta["lon"], "GFS") or []
        gfs_err = None
    except Exception as e:
        gfs_recs, gfs_err = [], f"{type(e).__name__}: {e}"

    sel = selection_payload(scope).get("selected", [])
    by_model = []
    for s in sel:
        try:
            rows = predict_for_cycle(
                CSC2_MODELS_DIR / scope / s["name"],
                buoy_id=buoy_id, euro_recs=euro_recs, gfs_recs=gfs_recs,
                cycle_utc=cycle_utc,
            )
            by_model.append({
                "name": s["name"], "arch": s["arch"],
                "is_top_performer": s["is_top_performer"],
                "composite_skill": s["composite_skill"],
                "rows": rows, "error": None,
            })
        except Exception as e:
            by_model.append({
                "name": s["name"], "arch": s["arch"],
                "is_top_performer": s["is_top_performer"],
                "composite_skill": s["composite_skill"],
                "rows": [], "error": f"{type(e).__name__}: {e}",
            })

    return {
        "buoy_id":      buoy_id,
        "buoy_label":   meta["label"],
        "cycle_utc":    cycle_utc,
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_euro_rows":  len(euro_recs),
        "n_gfs_rows":   len(gfs_recs),
        "euro_error":   euro_err,
        "gfs_error":    gfs_err,
        "models":       by_model,
    }


@app.route("/api/csc2/forecast")
def api_csc2_forecast():
    """Live CSC2 correction for one buoy. Cached 30 min via the helper above."""
    from csc2.schema import buoy_meta
    buoy_id = request.args.get("buoy_id", "44065")
    try:
        scope = buoy_meta(buoy_id)["scope"]
    except KeyError:
        return jsonify({"error": f"unknown buoy {buoy_id}"}), 400
    payload = _csc2_forecast_payload(buoy_id, scope)
    if payload is None:
        return jsonify({"error": "compute failed"}), 500
    return jsonify(payload)


@app.route("/palette-preview")
def palette_preview_page():
    """Static visual comparison of the current swell + wind palettes and
    five alternative wind palettes. Pick one (A–E) and ask to apply it —
    this page has no save action, it's for eyeballing only."""
    return render_template("palette-preview.html")


@app.route("/tuner")
def tuner_page():
    """Interactive slider-driven tuner for swell + wind category thresholds.
    Changes write back to the TOML files and trigger the same reload hook
    /api/refresh uses, so every downstream consumer picks them up live."""
    import wind_rules
    bands = swell_rules.load_bands()
    # Build a JSON-friendly copy of the swell bands — preserve 'always'/'never'
    # markers so the UI knows which cells are non-tunable.
    def _jsonable_rule(v):
        if isinstance(v, dict): return {"gte": v["gte"]}
        if isinstance(v, float): return v
        return v   # 'always' / 'never' strings
    swell_payload = {
        "bands": [
            {"period_ub": b["period_ub"],
             "rules": {k: _jsonable_rule(v) for k, v in b["rules"].items()}}
            for b in bands
        ]
    }
    # Light-mode palette for the tuner previews. Cell backgrounds come
    # straight from swell_rules.COLORS' light_bg field so the heatmap
    # matches what the main dashboard shows in light mode.
    cat_colors = {
        c: {"bg": swell_rules.COLORS[c]["light_bg"],
            "text": swell_rules.COLORS[c]["light_text"]}
        for c in swell_rules.CATEGORIES
    }
    wind_colors = {
        "Glassy":   {"bg": "#ccecd4", "text": "#166028"},
        "Groomed":  {"bg": "#ccecd4", "text": "#166028"},
        "Clean":    {"bg": "#ccecd4", "text": "#166028"},
        "Textured": {"bg": "#f5e6c0", "text": "#7a5500"},
        "Messy":    {"bg": "#d8e8f8", "text": "#1a5a9a"},
        "Blown Out":{"bg": "#e2e2de", "text": "#70707c"},
    }
    payload = {
        "swell": swell_payload,
        "wind":  wind_rules.load_config(),
        "categories":    swell_rules.CATEGORIES,
        "cat_colors":    cat_colors,
        "wind_ratings":  ["Glassy","Groomed","Clean","Textured","Messy","Blown Out"],
        "wind_colors":   wind_colors,
    }
    return render_template(
        "tuner.html",
        inline_config=_json.dumps(payload, separators=(',', ':')),
    )


@app.route("/api/tuner/save", methods=["POST"])
def api_tuner_save():
    """Persist swell + wind threshold edits to their TOML files and reload
    the in-memory caches. Any downstream endpoint picking up category
    classifications from `swell_rules` / `wind_rules` uses the new values
    on its next call."""
    payload = request.get_json(silent=True) or {}
    try:
        _write_swell_toml(payload.get("swell", {}))
        _write_wind_toml(payload.get("wind", {}))
        swell_rules.reload()
        import wind_rules
        wind_rules.reload()
        # Bust per-fetcher caches so any stale category labels get rebuilt
        _cache.clear_all()
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"status": "saved", "reloaded": ["swell_rules", "wind_rules"]})


def _render_rule(v):
    """Serialize one rule back to TOML syntax."""
    if isinstance(v, dict) and "gte" in v:
        return f'">={v["gte"]}"'
    if isinstance(v, (int, float)):
        return f"{float(v)}"
    # string ('always' | 'never') or fallback
    return f'"{v}"'


def _write_swell_toml(swell):
    bands = swell.get("bands") or []
    lines = [
        "# Swell Categorization Scheme",
        "# (auto-written by /api/tuner/save; edit in /tuner or this file)",
        "",
    ]
    for b in bands:
        ub = b.get("period_ub")
        ub_str = '"inf"' if ub is None else f"{float(ub)}"
        lines.append("[[band]]")
        lines.append(f"period_upper_bound = {ub_str}")
        rules = b.get("rules") or {}
        for cat in swell_rules.CATEGORIES:
            if cat not in rules:
                continue
            lines.append(f"{cat:<8}= {_render_rule(rules[cat])}")
        lines.append("")
    _atomic_write_text(
        os.path.join(app.root_path, "swell-categorization-scheme.toml"),
        "\n".join(lines) + "\n",
    )


def _write_wind_toml(wind):
    def g(p, default=0.0):
        cur = wind
        for k in p.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return float(cur)
    txt = f"""# Wind Categorization Scheme
# (auto-written by /api/tuner/save; edit in /tuner or this file)
# Sustained-wind-speed thresholds — matrix Y-axis on /tuner is authoritative.

[angles]
offshore_max  = {g('angles.offshore_max')}
sideshore_max = {g('angles.sideshore_max')}

[low_sustained]
clean_max = {g('low_sustained.clean_max')}

[offshore]
glassy_sust_max       = {g('offshore.glassy_sust_max')}
groomed_sustained_min = {g('offshore.groomed_sustained_min')}

[sideshore]
textured_sust_max = {g('sideshore.textured_sust_max')}
messy_sust_max    = {g('sideshore.messy_sust_max')}

[onshore]
textured_sust_max = {g('onshore.textured_sust_max')}
messy_sust_max    = {g('onshore.messy_sust_max')}
"""
    _atomic_write_text(
        os.path.join(app.root_path, "wind-categorization-scheme.toml"),
        txt,
    )


def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


@app.route("/favicon.svg")
def favicon():
    return send_from_directory(app.root_path, "favicon.svg", mimetype="image/svg+xml")

@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return send_from_directory(app.root_path, "apple-touch-icon.png", mimetype="image/png")


# ─── Cache-Control headers for Cloudflare edge caching ──────────────────────
@app.after_request
def _add_cache_headers(response):
    """Add Cache-Control headers so Cloudflare caches API responses at the edge.
    stale-while-revalidate: serve stale while refreshing in background.
    stale-if-error: serve stale if origin errors (e.g. during Open-Meteo outages).
    """
    if request.method != "GET" or response.status_code != 200:
        return response
    path = request.path

    if path == "/api/config":
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=3600, stale-while-revalidate=7200, stale-if-error=86400"
    elif path == "/api/sun":
        response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=43200, stale-while-revalidate=43200"
    elif path.startswith("/api/forecast/"):
        response.headers["Cache-Control"] = "public, max-age=120, s-maxage=1800, stale-while-revalidate=7200, stale-if-error=14400"
    elif path in ("/api/wind", "/api/wind_forecast", "/api/region_wind"):
        response.headers["Cache-Control"] = "public, max-age=120, s-maxage=1800, stale-while-revalidate=3600, stale-if-error=7200"
    elif path == "/api/buoys":
        # No stale-while-revalidate: SWR caused reloads to serve stale buoy
        # readings while a background revalidation ran, so the *next* reload
        # still saw stale data. Short max-age with no SWR → every reload past
        # the max-age window pulls fresh readings synchronously from origin.
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=60, stale-if-error=1800"
    elif path == "/api/tides":
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=7200, stale-while-revalidate=14400, stale-if-error=86400"
    elif path.startswith("/api/buoy_history/") or path == "/api/buoy_historical_context":
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=1800, stale-while-revalidate=3600, stale-if-error=14400"

    return response


# ─── Background cache warming ─────────────────────────────────────────────────
_WARM_INTERVAL = 1800   # 30 minutes — well within TTL of 3600s


def _warm_all_caches():
    """Pre-fetch all data so user requests always hit warm cache."""
    t0 = time.monotonic()
    errors = []

    # Wave forecasts.
    # GFS: batched through Open-Meteo (1 API call, all spots).
    # EURO: CMEMS, parallel across region buoys (~11 s for 7 buoys in testing).
    try:
        fetch_all_wave_forecasts("GFS")
    except Exception as e:
        errors.append(f"wave/GFS: {e}")
    try:
        fetch_all_cmems_wave_forecasts()
    except Exception as e:
        errors.append(f"cmems: {e}")

    # Wind (already batched internally)
    for model in ("EURO", "GFS"):
        try:
            fetch_wind_grid(model)
        except Exception as e:
            errors.append(f"wind_grid/{model}: {e}")
        try:
            fetch_wind_forecast_grid(model)
        except Exception as e:
            errors.append(f"wind_forecast/{model}: {e}")

    # Region wind
    for model in ("EURO", "GFS"):
        try:
            fetch_region_wind_forecasts(model)
        except Exception as e:
            errors.append(f"region_wind/{model}: {e}")

    # Buoys (parallel)
    try:
        futures = {_buoy_pool.submit(fetch_buoy, s["buoy_id"]): s["name"] for s in SPOTS}
        for f in as_completed(futures, timeout=30):
            f.result()
    except Exception as e:
        errors.append(f"buoys: {e}")

    # Tides
    try:
        fetch_tide_predictions()
    except Exception as e:
        errors.append(f"tides: {e}")

    # CSC2 forecast — pre-compute predictions for every east buoy so the
    # /csc page never pays the cold cost. Skipped if no models are trained.
    try:
        from csc2.schema import buoys_in as _csc2_buoys_in
        from csc2.registry import list_models as _csc2_list_models
        if _csc2_list_models("east"):
            for _bid in _csc2_buoys_in("east"):
                try:
                    _csc2_forecast_payload(_bid, "east")
                except Exception as e:
                    errors.append(f"csc2_forecast/{_bid}: {e}")
    except Exception as e:
        errors.append(f"csc2_forecast: {e}")

    elapsed = time.monotonic() - t0
    if errors:
        print(f"[cache-warm] done in {elapsed:.1f}s with {len(errors)} errors: "
              + "; ".join(errors))
    else:
        print(f"[cache-warm] all caches refreshed in {elapsed:.1f}s")


def _cache_warmer_loop():
    """Background thread: warm caches on startup and then every WARM_INTERVAL seconds."""
    # Initial warm on startup (wait a few seconds for Flask to be ready)
    time.sleep(3)
    print("[cache-warm] initial cache warm starting…")
    _warm_all_caches()

    while True:
        time.sleep(_WARM_INTERVAL)
        try:
            _warm_all_caches()
        except Exception as e:
            print(f"[cache-warm] loop error: {e}")


if __name__ == "__main__":
    import socket
    def _get_local_ip():
        """Get the LAN IP by opening a UDP socket (no traffic sent)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "<your-local-ip>"
    _local_ip = _get_local_ip()
    mode = "development" if DEBUG_MODE else "production"
    print(f"\n  ◈ colesurfs ({mode})")
    print(f"  ─────────────────────────────────")
    print(f"  Host    → {HOST}:{PORT}")
    if HOST == "0.0.0.0":
        print(f"  Local   → http://127.0.0.1:{PORT}")
        print(f"  Network → http://{_local_ip}:{PORT}")
    else:
        print(f"  Local   → http://{HOST}:{PORT}")
    print(f"  Debug   → {'on' if DEBUG_MODE else 'off'}")
    print(f"  Warmer  → every {_WARM_INTERVAL}s")
    print(f"  Press Ctrl+C to stop.\n")

    # Start background cache warmer
    _warmer = threading.Thread(target=_cache_warmer_loop, daemon=True)
    _warmer.start()

    serve(app, host=HOST, port=PORT, threads=8)
