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
  /api/status?model=             Model run estimate + daily API usage
  /api/debug/spectral/<id>       Diagnostic: raw spectral parse (COLESURFS_DEBUG=1 only)
  /api/buoy_history/<station_id>  5-day historical buoy data with spectral components
  /api/refresh (POST)            Clear caches + reload swell rules
"""
import os
import time
import threading
from flask import Flask, jsonify, render_template, request, send_from_directory
from waitress import serve

from config import SPOTS, WIND_SPOTS, MODEL_COLORS, WIND_BANDS, REGION_VIEWS
import swell_rules
from buoy  import fetch_buoy, fetch_buoy_history, _parse_spectral_file, _spectral_components
import requests as _req_buoy
from waves import fetch_wave_forecast
from wind  import (fetch_wind_grid, fetch_spot_wind,
                    fetch_wind_forecast_grid, fetch_spot_wind_forecasts,
                    fetch_region_wind_forecasts, estimate_model_run)
from tide  import fetch_tide_predictions
import cache as _cache

app = Flask(__name__)
PORT = int(os.environ.get("COLESURFS_PORT", 5151))
HOST = os.environ.get("COLESURFS_HOST", "127.0.0.1")
DEBUG_MODE = os.environ.get("COLESURFS_DEBUG", "").strip() == "1"


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


@app.route("/api/buoys")
def api_buoys():
    return jsonify({s["name"]: fetch_buoy(s["buoy_id"]) for s in SPOTS})


@app.route("/api/forecast/<model_key>")
def api_forecast(model_key: str):
    if model_key not in ("EURO", "GFS"):
        return jsonify({"error": "unknown model"}), 400
    return jsonify({
        s["name"]: fetch_wave_forecast(s["lat"], s["lon"], model_key)
        for s in SPOTS
    })


@app.route("/api/wind")
def api_wind():
    model_key = request.args.get("model", "EURO").upper()
    if model_key not in ("EURO", "GFS"):
        model_key = "EURO"
    return jsonify({
        "grid":       fetch_wind_grid(model_key),
        "spot_winds": {s["name"]: fetch_spot_wind(s["lat"], s["lon"]) for s in SPOTS},
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
    """Hourly wind + gust forecast for all WIND_SPOTS — for Regional Mode table."""
    model_key = request.args.get("model", "EURO").upper()
    if model_key not in ("EURO", "GFS"):
        model_key = "EURO"
    return jsonify(fetch_region_wind_forecasts(model_key))


@app.route("/api/tides")
def api_tides():
    """Hourly tide predictions (height ft + daily %) for all WIND_SPOT tide stations."""
    return jsonify(fetch_tide_predictions())


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


@app.route("/api/config")
def api_config():
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
    """Clear all caches and reload swell rules. Rate-limited to 1 call per 30s."""
    ip = request.remote_addr or "unknown"
    if _rate_limited("refresh", ip, max_calls=1, window_sec=30):
        return jsonify({"error": "rate limit exceeded, try again in 30s"}), 429
    _cache.clear_all()
    swell_rules.reload()
    return jsonify({"status": "cache cleared, swell rules reloaded"})


@app.route("/api/buoy_history/<station_id>")
def api_buoy_history(station_id):
    """5-day historical buoy data with spectral swell components."""
    valid_ids = {s["buoy_id"] for s in SPOTS}
    if station_id not in valid_ids:
        return jsonify({"error": "unknown station"}), 404
    data = fetch_buoy_history(station_id)
    if data is None:
        return jsonify({"error": "data unavailable"}), 503
    return jsonify(data)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(app.root_path, "favicon.svg", mimetype="image/svg+xml")

@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return send_from_directory(app.root_path, "apple-touch-icon.png", mimetype="image/png")


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
    print(f"  Press Ctrl+C to stop.\n")
    serve(app, host=HOST, port=PORT, threads=8)
