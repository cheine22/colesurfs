"""
colesurfs — Flask Application

Routes:
  /                              Dashboard UI (single-page app)
  /review, /seasons, /csc, /csc-model, /gland, /tuner, /gland/tuner, /palette-preview
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
  /api/fun_days                  Observed fun+ ledger per buoy (fun_days.py)
  /api/review, /api/review/seasons   Ledger rows for /review + season tables for /seasons
  /api/refresh (POST)            Clear caches + reload swell rules
  /api/tuner/save, /api/gland/tuner/save (POST)   Write the TOML schemes
  /api/gland/*                   G-Land page data (gland.py)
  /api/csc2/*                    CSC2 archive status, model registry, live correction
  /tiles/bathy/<style>/<theme>/z/x/y.png   Self-rendered basemap (bathy.py)

EURO waves come from Copernicus Marine (CMEMS) ECMWF-WAM ANFC since v1.5;
GFS waves from Open-Meteo.
"""
import ipaddress as _ipaddress
import json as _json
import os
import tempfile as _tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from flask_compress import Compress
from waitress import serve

from config import SPOTS, WIND_SPOTS, MODEL_COLORS, WIND_BANDS, REGION_VIEWS
import swell_rules
import wind_rules
from buoy  import (fetch_buoy, fetch_buoy_history, fetch_buoy_historical_context,
                    _parse_spectral_file, _spectral_components)
import requests as _req_buoy
from waves import fetch_wave_forecast, fetch_all_wave_forecasts
from waves_cmems import fetch_all_cmems_wave_forecasts
from wind  import (fetch_wind_grid, fetch_spot_wind, fetch_all_spot_winds,
                    fetch_wind_forecast_grid, fetch_spot_wind_forecasts,
                    fetch_region_wind_forecasts, estimate_model_run)
from tide  import fetch_tide_predictions
from sun   import compute_sun_data
from fun_days import (all_summaries as _fun_days_all, review_payload as _review_payload,
                      season_tables as _season_tables)
import cache as _cache
import bathy
import numpy as np

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


def _client_ip() -> str:
    """Real client IP for rate-limit bucketing. Public traffic arrives via the
    Cloudflare tunnel over loopback, so remote_addr alone would put every
    visitor in one shared bucket; prefer the CF-set header when present."""
    return request.headers.get("CF-Connecting-IP") or request.remote_addr or "unknown"


def _rate_limited(bucket: str, ip: str, max_calls: int, window_sec: int) -> bool:
    """Return True if this IP has exceeded max_calls in the last window_sec seconds."""
    key = f"{bucket}:{ip}"
    now = time.monotonic()
    with _rate_lock:
        # Prune dead keys so the dict doesn't grow one entry per unique IP forever.
        if len(_rate_hits) > 512:
            for k in [k for k, v in _rate_hits.items() if not v or now - v[-1] > 3600]:
                del _rate_hits[k]
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
    """General rate limit on /api/* routes. A single dashboard page load
    legitimately fires ~17-19 /api requests (both forecasts, both models'
    region winds, wind grid + forecast, tides, buoys, sun, and one
    historical-context call per buoy), so the ceiling must clear several
    rapid refreshes. At 60/min the ~4th reload tripped it and left the page
    half-loaded ("repeated refreshes disrupt the site"); 240/min ≈ 12 full
    reloads per minute while still capping scrapers / runaway clients."""
    if request.path.startswith("/api/"):
        if _rate_limited("api", _client_ip(), max_calls=240, window_sec=60):
            resp = jsonify({"error": "rate limit exceeded"})
            resp.headers["Retry-After"] = "30"
            return resp, 429


@app.before_request
def _restrict_tuner():
    """Gate /tuner + /api/tuner/* to LAN clients only. The Cloudflare tunnel
    proxies public requests through loopback, so we can't rely on remote_addr
    alone — we additionally reject any request carrying Cloudflare-set
    headers (CF-Ray, CF-Connecting-IP) or one where the Host header isn't
    a LAN/loopback address."""
    path = request.path or ""
    if (path not in ("/tuner", "/gland/tuner")
            and not path.startswith("/api/tuner/")
            and not path.startswith("/api/gland/tuner/")):
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


# Last-known-good fallback per forecast model. Whenever the upstream
# fetch returns a populated dict, we stash it here. If a subsequent
# fetch comes back empty (transient cache-wipe gap, brief upstream
# blip, slow first cold fetch), we serve this stale copy instead of
# returning {}, so the dashboard's outage modal only fires when we've
# *never* had usable data — i.e. genuine upstream failure with no
# history to fall back on. v1.9: persisted to disk so the fallback
# also covers the first requests after a process restart.
_LKG_PATH = Path(__file__).parent / ".cache" / "lkg_forecast.json"


def _load_lkg() -> dict[str, dict]:
    try:
        with open(_LKG_PATH) as f:
            d = _json.load(f)
        return {"EURO": d.get("EURO") or {}, "GFS": d.get("GFS") or {}}
    except Exception:
        return {"EURO": {}, "GFS": {}}


_last_known_forecast: dict[str, dict] = _load_lkg()
# 2026-09-19: serialised. The frontend requests /api/forecast/EURO and /GFS
# together, so after a TTL rollover two threads wrote the same .tmp at once
# and os.replace promoted a spliced, unparseable file (found 2026-09-18).
_lkg_lock = threading.Lock()


def _stash_lkg(model: str, fresh: dict) -> None:
    # `fresh` is the TTL-cached object, so identity comparison suffices to
    # skip redundant disk writes on every request within a cache window.
    if fresh is _last_known_forecast.get(model):
        return
    with _lkg_lock:
        _last_known_forecast[model] = fresh
        # 2026-09-23: per-writer temp file. The splice recurred on 09-23 with
        # the lock loaded, so a second process shares .cache/; a fixed ".tmp"
        # name lets two processes interleave into one file before os.replace.
        tmp = None
        try:
            _LKG_PATH.parent.mkdir(exist_ok=True)
            fd, tmp = _tempfile.mkstemp(dir=_LKG_PATH.parent,
                                        prefix=_LKG_PATH.name + ".", suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                _json.dump(_last_known_forecast, f, separators=(',', ':'))
            os.replace(tmp, _LKG_PATH)
        except Exception as e:
            print(f"[lkg] persist failed: {type(e).__name__}: {e}")
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


def _is_populated(d: dict | None) -> bool:
    """True if d is a dict with at least one non-null spot value."""
    if not d:
        return False
    return any(v is not None for v in d.values())


def _with_status(payload: dict) -> dict:
    """Attach `_status: "partial"` when some (but not all) spots are null,
    so the frontend can badge degraded data instead of silently gapping.
    Returns a copy — never mutates the TTL-cached object."""
    vals = [v for k, v in payload.items() if not k.startswith("_")]
    if vals and any(v is None for v in vals) and any(v is not None for v in vals):
        return {**payload, "_status": "partial"}
    return payload


@app.route("/api/forecast/<model_key>")
def api_forecast(model_key: str):
    """Model keys:
      EURO — Copernicus Marine ECMWF WAM ANFC (per-buoy 3h, interpolated hourly,
             SW1/SW2 partitions with Tm01→Tp scaled periods)
      GFS  — Open-Meteo NCEP GFS-Wave (per-spot hourly)

    v1.5: Open-Meteo ECMWF-WAM was removed from the site; EURO is now CMEMS
    exclusively. /api/forecast/C-EURO remains as a backward-compat alias.
    v1.8: Last-known-good fallback added — if a fresh fetch comes back
    empty, we return the previous successful response instead of {}.
    """
    key = model_key.upper()
    if key in ("EURO", "C-EURO"):
        key = "EURO"
        fresh = fetch_all_cmems_wave_forecasts()
    elif key == "GFS":
        fresh = fetch_all_wave_forecasts("GFS")
    else:
        return jsonify({"error": "unknown model"}), 400

    if _is_populated(fresh):
        _stash_lkg(key, fresh)
        return jsonify(_with_status(fresh))
    stale = _last_known_forecast.get(key) or {}
    if stale:
        return jsonify({**stale, "_status": "stale"})
    return jsonify({})


@app.route("/api/wind")
def api_wind():
    model_key = request.args.get("model", "EURO").upper()
    if model_key not in ("EURO", "GFS"):
        model_key = "EURO"
    # One batched request for all spots; fall back to parallel per-spot
    # fetches (each independently cached) only if the batch fails.
    spot_winds = fetch_all_spot_winds()
    if spot_winds is None:
        spot_futures = {
            _buoy_pool.submit(fetch_spot_wind, s["lat"], s["lon"]): s["name"]
            for s in SPOTS
        }
        spot_winds = {name: None for name in spot_futures.values()}
        try:
            for future in as_completed(spot_futures, timeout=8):
                name = spot_futures[future]
                try:
                    spot_winds[name] = future.result(timeout=5)
                except Exception:
                    spot_winds[name] = None
        except TimeoutError:
            pass   # serve whatever completed; missing spots stay None
    grid = fetch_wind_grid(model_key)
    payload = {"grid": grid, "spot_winds": spot_winds}
    if grid is None or any(v is None for v in spot_winds.values()):
        payload["_status"] = "partial"
    return jsonify(payload)


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


def _config_payload() -> dict:
    """Single source for the config object served by /api/config and inlined
    into index.html. Rebuilt per call so /tuner saves are picked up live."""
    bands = swell_rules.load_bands()
    return {
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


@app.route("/api/config")
def api_config():
    return jsonify(_config_payload())


@app.route("/tiles/bathy/<style>/<theme>/<int:z>/<int:x>/<int:y>.png")
def bathy_tile(style, theme, z, x, y):
    """Self-rendered basemap (see bathy.py). Immutable per style version."""
    if style != bathy.STYLE:
        abort(404)
    try:
        png = bathy.tile_png(theme, z, x, y)
    except Exception as e:
        print(f"[bathy] {z}/{x}/{y} failed: {e}", flush=True)
        abort(502)
    if png is None:
        abort(404)
    resp = Response(png, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp


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
    if _rate_limited("refresh", _client_ip(), max_calls=1, window_sec=30):
        return jsonify({"error": "rate limit exceeded, try again in 30s"}), 429
    _cache.clear_all()
    swell_rules.reload()
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


@app.route("/api/fun_days")
def api_fun_days():
    """Observed fun+ ledger per buoy: days since the last fun+ day and this
    calendar year's tally by category (fun_days.py). Keyed by buoy_id."""
    data = _fun_days_all()
    if data is None:
        return jsonify({"error": "data unavailable"}), 503
    return jsonify(data)


# ─── Home-screen widget ───────────────────────────────────────────────────────
# Server-side port of index.html's computeModelOverview (the Fun+ Days cell)
# so a phone widget can read the same number the dashboard shows. Any change
# to that JS rule must be mirrored here — the two are meant to agree exactly.
_WIDGET_SITE = "https://surfreport.coleheine.com/"
_SURFABLE_WIND = {"Glassy", "Groomed", "Clean", "Textured"}


def _region_surfable_hours(region: str, wind: dict | None) -> set:
    """Hours ('YYYY-MM-DDTHH:MM') where ≥1 wind spot in the region rates
    Textured-or-better — the Fun+ Days gate (_regionCleanWind().surfable)."""
    ok = set()
    if not wind:
        return ok
    for ws in WIND_SPOTS:
        if ws.get("buoy_region") != region or ws.get("shore_normal") is None:
            continue
        for rec in wind.get(ws["name"]) or []:
            if not rec or rec.get("speed_mph") is None or rec["time"] in ok:
                continue
            cond = wind_rules.categorize(rec["speed_mph"], rec.get("direction_deg"),
                                         ws["shore_normal"], rec.get("gust_mph"))
            if cond in _SURFABLE_WIND:
                ok.add(rec["time"])
    return ok


def _widget_overview(region: str, euro: dict, gfs: dict, wind: dict | None,
                     sun: dict, now) -> dict:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from config import TIMEZONE
    tz = ZoneInfo(TIMEZONE)
    cats = swell_rules.CATEGORIES
    fun_idx = cats.index("FUN")

    def _hp(rec):
        if not rec or rec.get("wave_height_ft") is None:
            return None
        comps = rec.get("components") or []
        return (comps[0]["height_ft"], comps[0]["period_s"]) if comps \
            else (rec["wave_height_ft"], rec["wave_period_s"])

    def _ms(t):   # local wall-clock string → epoch ms, as JS `new Date('Y-m-d H:M')`
        return datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=tz).timestamp() * 1000

    e_by_t = {r["time"]: r for r in (euro.get(region) or [])}
    g_by_t = {r["time"]: r for r in (gfs.get(region) or [])}
    now_floor = now.astimezone(tz).replace(minute=0, second=0, microsecond=0)
    times = sorted(t for t in set(e_by_t) | set(g_by_t) if _ms(t) >= now_floor.timestamp() * 1000)
    sampled = []
    if times:
        t0 = _ms(times[0])
        sampled = [t for t in times if ((_ms(t) - t0) / 3600000) % 3 == 0]

    surfable = _region_surfable_hours(region, wind)
    apply_wind = len(surfable) > 0          # honest-empty: no wind → swell only
    three_h = 3 * 3600 * 1000
    thirty = 30 * 60 * 1000
    per_day, best_idx, best_hp = {}, -1, None
    for t in sampled:
        s = sun.get(t[:10])
        if s:
            t_ms = _ms(t)
            if t_ms + three_h <= _ms(s["sunrise"]) - thirty or t_ms >= _ms(s["sunset"]) + thirty:
                continue
        e_hp, g_hp = _hp(e_by_t.get(t)), _hp(g_by_t.get(t))
        if not e_hp or not g_hp:
            continue
        ei = cats.index(swell_rules.categorize(*e_hp))
        gi = cats.index(swell_rules.categorize(*g_hp))
        m = min(ei, gi)
        if m > best_idx:
            best_idx, best_hp = m, (e_hp if ei <= gi else g_hp)
        if m >= fun_idx:
            if apply_wind and t not in surfable:
                continue
            per_day[t[:10]] = per_day.get(t[:10], 0) + 1
    count = sum(1 for v in per_day.values() if v >= 2)
    window_days = round((_ms(sampled[-1]) - _ms(sampled[0])) / 86400000) if len(sampled) >= 2 else 0
    cat = cats[best_idx] if best_idx >= 0 else None
    return {
        "count": count,
        "window_days": window_days,
        "category": cat,
        "colors": dict(swell_rules.COLORS[cat]) if cat else None,
        "best": {"height_ft": best_hp[0], "period_s": best_hp[1]} if best_hp else None,
        "fun_days": sorted(d for d, v in per_day.items() if v >= 2),
        "wind_gated": apply_wind,
    }


def _widget_regions_arg():
    """(names, error_response) from `regions=` — comma-separated region names,
    default every dashboard region in regions.yaml order."""
    names = [n.strip() for n in (request.args.get("regions") or "").split(",") if n.strip()]
    known = {s["name"]: s for s in SPOTS}
    if not names:
        names = list(known)
    bad = [n for n in names if n not in known]
    if bad:
        return None, (jsonify({"error": f"unknown region(s): {', '.join(bad)}",
                               "regions": list(known)}), 400)
    return names, None


def _widget_payload(names: list[str]) -> dict:
    from datetime import datetime, timezone as _tz
    known = {s["name"]: s for s in SPOTS}
    euro = fetch_all_cmems_wave_forecasts() or _last_known_forecast.get("EURO") or {}
    gfs = fetch_all_wave_forecasts("GFS") or _last_known_forecast.get("GFS") or {}
    try:
        wind = fetch_region_wind_forecasts("EURO")
    except Exception:
        wind = None
    spot = SPOTS[0] if SPOTS else {"lat": 40.58, "lon": -73.63}
    sun = compute_sun_data(spot["lat"], spot["lon"])
    now = datetime.now(_tz.utc)

    runs = {k: estimate_model_run(k) for k in ("EURO", "GFS")}
    oldest = min(runs.values(), key=lambda r: r.get("run_time") or "")
    rt = oldest.get("run_time") or ""
    # "today 12Z" / "yesterday 0Z": the run's day in local time (a 00Z run is
    # the previous evening here), hour without the leading zero.
    label = "—"
    if len(rt) >= 16:
        from zoneinfo import ZoneInfo
        from config import TIMEZONE
        run_local = datetime.strptime(rt, "%Y-%m-%dT%H:%MZ").replace(tzinfo=_tz.utc).astimezone(ZoneInfo(TIMEZONE))
        days_ago = (now.astimezone(ZoneInfo(TIMEZONE)).date() - run_local.date()).days
        day = "today" if days_ago == 0 else "yesterday" if days_ago == 1 else f"{rt[5:7]}/{rt[8:10]}"
        label = f"{day} {int(rt[11:13])}Z"

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "site": _WIDGET_SITE,
        "last_update": {"model": oldest.get("model"), "run_time": rt, "label": label},
        "runs": {k: {"run_time": v.get("run_time"), "run_utc": v.get("run_utc")} for k, v in runs.items()},
        "regions": [{"name": n, "buoy_id": known[n]["buoy_id"],
                     **_widget_overview(n, euro, gfs, wind, sun, now)} for n in names],
    }


@app.route("/api/widget")
def api_widget():
    """Fun+ Days per region for the iOS widget. `regions` = comma-separated
    region names (default: every dashboard region, in regions.yaml order).
    `last_update` is the OLDER of the two model runs, so the stamp never
    claims freshness one model lacks."""
    names, err = _widget_regions_arg()
    if err:
        return err
    return jsonify(_widget_payload(names))


# ── live-lido widget ─────────────────────────────────────────────────────────
# Medium widget, Scriptable parameters live-lido / live-landing /
# live-southampton: the spot's buoy now (left, swell tint) and the spot's tide /
# wind (right, wind tint). Any WIND_SPOTS entry with a tide station and a shore
# normal works; the buoy is the spot's buoy_region. The trend rules below are
# the ones the mockup page implements in JS.
_LIVE_DEFAULT_SPOT = "Lido Beach"


def _live_spot_arg():
    """(spot dict, error response) from `spot=` (default Lido Beach)."""
    name = (request.args.get("spot") or _LIVE_DEFAULT_SPOT).strip()
    spot = next((ws for ws in WIND_SPOTS if ws["name"].lower() == name.lower()), None)
    if not spot or not spot.get("tide_station") or spot.get("shore_normal") is None:
        ok = [ws["name"] for ws in WIND_SPOTS if ws.get("tide_station") and ws.get("shore_normal") is not None]
        return None, (jsonify({"error": f"unknown live spot: {name}", "spots": ok}), 400)
    return spot, None
_WIND_GOOD = {"Glassy", "Groomed", "Clean"}
_WIND_RANK = {"Glassy": 0, "Groomed": 1, "Clean": 2, "Textured": 3, "Messy": 4, "Blown Out": 5}


def _wind_tier(c):
    return "good" if c in _WIND_GOOD else "mid" if c == "Textured" else "bad"


def _wind_better(a, b):
    return a if _WIND_RANK.get(a, 9) <= _WIND_RANK.get(b, 9) else b


def _wind_worse(a, b):
    return a if _WIND_RANK.get(a, -1) >= _WIND_RANK.get(b, -1) else b


def _fmt_hour(h: int) -> str:
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def _fmt_clock(dt) -> str:
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def _live_wind_sentence(now: dict, ahead: list, after_sunset: bool) -> str:
    """now = {t, e, g} (EURO / GFS rating at Lido this hour), ahead = the hours after it."""
    cur = _wind_better(now["e"], now["g"])
    t0 = _wind_tier(cur)
    at = lambda h: f"at {_fmt_hour(h)}{' tomorrow' if after_sunset else ''}"
    if t0 != "good":
        for w in ahead:                      # improving: first hour EITHER model rates good
            b = _wind_better(w["e"], w["g"])
            if b in _WIND_GOOD:
                return f"Wind trending {b.lower()} {at(w['t'])}"
        if t0 == "bad":
            for w in ahead:
                if _wind_tier(_wind_better(w["e"], w["g"])) == "mid":
                    return f"Wind trending textured {at(w['t'])}"
    else:
        for w in ahead:                      # worsening: first hour EITHER model leaves good
            x = _wind_worse(w["e"], w["g"])
            if x not in _WIND_GOOD:
                return (f"Wind deteriorating {at(w['t'])}" if _wind_tier(x) == "bad"
                        else f"Wind trending textured {at(w['t'])}")
    return "Wind holding tomorrow" if after_sunset else "Wind holding for rest of day"


def _live_payload(spot: dict) -> dict:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from config import TIMEZONE
    import math
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    today = now.date()
    _LIVE_BUOY_REGION = spot["buoy_region"]
    _LIVE_SPOT = spot["name"]
    region = next(sp for sp in SPOTS if sp["name"] == _LIVE_BUOY_REGION)
    cats = swell_rules.CATEGORIES

    # ── buoy ──
    raw = fetch_buoy(region["buoy_id"]) or {}
    comps = [c for c in (raw.get("components") or []) if c.get("height_ft") is not None]
    if not comps and raw.get("wave_height_ft") is not None:
        comps = [{"height_ft": raw["wave_height_ft"], "period_s": raw.get("wave_period_s"),
                  "direction_deg": raw.get("wave_direction_deg"), "type": "swell"}]
    primary = comps[0] if comps else None
    buoy_cat = swell_rules.categorize(primary["height_ft"], primary["period_s"]) if primary and primary.get("period_s") else None
    try:
        obs_time = _fmt_clock(datetime.fromisoformat(raw["timestamp"]).astimezone(tz))
    except Exception:
        obs_time = None

    # ── sun / window: the rest of today's daylight, or tomorrow's (dawn patrol) after sunset ──
    sun = compute_sun_data(spot["lat"], spot["lon"], days=2)
    def _sun(d):
        r = sun.get(d.isoformat()) or {}
        p = lambda k: datetime.strptime(r[k], "%Y-%m-%dT%H:%M").replace(tzinfo=tz) if r.get(k) else None
        return p("sunrise"), p("sunset")
    sr_today, ss_today = _sun(today)
    if ss_today and now > ss_today:
        after_sunset, day = True, today + timedelta(days=1)
    elif sr_today and now < sr_today:
        after_sunset, day = True, today          # pre-dawn: today's dawn patrol
    else:
        after_sunset, day = False, today
    sr, ss = _sun(day)
    sunset_hour = ss.hour if ss else 18
    now_hour = (math.ceil(sr.hour + sr.minute / 60) if sr else 7) if after_sunset else now.hour
    hours = list(range(now_hour, sunset_hour + 1))
    key = lambda h: f"{day.isoformat()}T{h:02d}:00"

    # ── wind ratings per hour, both models ──
    def _ratings(model):
        try:
            recs = (fetch_region_wind_forecasts(model) or {}).get(_LIVE_SPOT) or []
        except Exception:
            recs = []
        return {r["time"]: wind_rules.categorize(r["speed_mph"], r.get("direction_deg"),
                                                 spot["shore_normal"], r.get("gust_mph"))
                for r in recs if r.get("speed_mph") is not None}
    we, wg = _ratings("EURO"), _ratings("GFS")
    wind_hours = [w for w in ({"t": h, "e": we.get(key(h)), "g": wg.get(key(h))} for h in hours) if w["e"] and w["g"]]
    wind_now = wind_hours[0] if wind_hours and wind_hours[0]["t"] == now_hour else None
    ahead = [w for w in wind_hours if w["t"] > now_hour]
    wind_sentence = _live_wind_sentence(wind_now, ahead, after_sunset) if wind_now else None
    wind_tint_cat = _wind_better(wind_now["e"], wind_now["g"]) if wind_now else None

    # ── swell trend: worst primary rating either model forecasts in the window, floored at the buoy ──
    euro = fetch_all_cmems_wave_forecasts() or _last_known_forecast.get("EURO") or {}
    gfs = fetch_all_wave_forecasts("GFS") or _last_known_forecast.get("GFS") or {}
    def _cat(rec):
        c = ((rec or {}).get("components") or [None])[0]
        return swell_rules.categorize(c["height_ft"], c["period_s"]) if c and c.get("height_ft") is not None and c.get("period_s") else None
    e_by = {r["time"]: r for r in (euro.get(_LIVE_BUOY_REGION) or [])}
    g_by = {r["time"]: r for r in (gfs.get(_LIVE_BUOY_REGION) or [])}
    swell_hours = [{"t": h, "e": _cat(e_by.get(key(h))), "g": _cat(g_by.get(key(h)))}
                   for h in hours if after_sunset or h > now_hour]
    worst = cats.index(buoy_cat) if buoy_cat else None
    for h in swell_hours:
        for m in ("e", "g"):
            if h[m]:
                worst = cats.index(h[m]) if worst is None else min(worst, cats.index(h[m]))
    # "staying" when the worst forecast rating matches the buoy's current one
    swell_sentence = (None if worst is None else
                      f"Swell staying {cats[worst].lower()}" if buoy_cat and worst == cats.index(buoy_cat)
                      else f"Swell trending {cats[worst].lower()}")

    # ── tide: today's curve, its highs and lows, the height now ──
    tides = (fetch_tide_predictions() or {}).get(_LIVE_SPOT) or {}
    midnight = datetime.combine(today, datetime.min.time())
    hourly = [tides.get((midnight + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M"), {}).get("height_ft") for h in range(25)]
    for i, v in enumerate(hourly):
        if v is None:
            hourly[i] = hourly[i - 1] if i else 0.0
    hilo = []
    for slot, rec in tides.items():
        if not rec.get("hilo_type"):
            continue
        iso = rec.get("hilo_iso")
        if not iso:
            # tide payload cached before hilo_iso existed: rebuild from the label
            # (an event stamped on a 00:00 slot that reads "pm" belongs to the day before)
            try:
                hm = datetime.strptime(rec["hilo_time"], "%I:%M%p")
            except (KeyError, ValueError):
                continue
            ev_day = datetime.strptime(slot[:10], "%Y-%m-%d").date()
            if slot[11:13] == "00" and hm.hour >= 12:
                ev_day -= timedelta(days=1)
            iso = f"{ev_day.isoformat()}T{hm.hour:02d}:{hm.minute:02d}"
        if not iso.startswith(today.isoformat()):
            continue
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M")
        hilo.append({"t": dt.hour + dt.minute / 60, "ft": rec.get("hilo_height_ft", rec.get("height_ft")),
                     "k": rec["hilo_type"], "lbl": rec.get("hilo_time", "").replace("am", "a").replace("pm", "p")})
    hilo.sort(key=lambda x: x["t"])
    now_h = now.hour + now.minute / 60
    i = min(now.hour, 23)
    frac = now_h - i
    tide_now = round(hourly[i] + (hourly[i + 1] - hourly[i]) * frac, 1)
    later = hourly[i] + (hourly[i + 1] - hourly[i]) * min(1.0, frac + 0.5)
    tide_trend = "rising" if later >= tide_now else "falling"

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "site": _WIDGET_SITE,
        "buoy": {"region": _LIVE_BUOY_REGION, "buoy_id": region["buoy_id"], "time": obs_time,
                 "category": buoy_cat, "colors": dict(swell_rules.COLORS[buoy_cat]) if buoy_cat else None,
                 "primary": primary, "secondary": comps[1] if len(comps) > 1 else None,
                 "swell_sentence": swell_sentence},
        "tap_url": spot.get("surfline_url") or _WIDGET_SITE,
        "lido": {"spot": _LIVE_SPOT, "after_sunset": after_sunset, "window_day": day.isoformat(),
                 "sunrise": _fmt_clock(sr) if sr else None, "sunset": _fmt_clock(ss) if ss else None,
                 "wind_now": wind_now, "wind_hours": wind_hours, "wind_sentence": wind_sentence,
                 "wind_category": wind_tint_cat, "swell_hours": swell_hours,
                 "tide": {"now_ft": tide_now, "trend": tide_trend, "now_h": round(now_h, 2),
                          "hourly": hourly, "hilo": hilo}},
    }


@app.route("/api/widget/live")
def api_widget_live():
    """`spot=` (default Lido Beach) picks the tide/wind spot; the buoy is its region's."""
    spot, err = _live_spot_arg()
    if err:
        return err
    return jsonify(_live_payload(spot))


# ── widget image ─────────────────────────────────────────────────────────────
# The widget mockup is the spec. Scriptable can't load the fonts or draw the
# glass, so the widget shows a PNG of templates/widget_render.html (the
# mockup's CSS verbatim) rendered by headless Chrome at the mockup's own
# dimensions and 3× scale; iOS scales it to the widget frame and rounds the
# corners. PNGs are cached by content hash under .cache/widget_png/.
_WIDGET_DIMS = {"small": (170, 170), "medium": (360, 170), "large": (360, 376)}
_WIDGET_N = {"small": 1, "medium": 2, "large": 4}
_WIDGET_PNG_DIR = Path(__file__).resolve().parent / ".cache" / "widget_png"
_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
_FLAT_DARK_INK = "#7a7a95"   # FLAT's cell ink vanishes on a widget; the mockup lifts it
_widget_render_lock = threading.Lock()   # single-flight: a widget's tiles arrive as 8–16 parallel requests


_FLAT_LIGHT_INK = "#52525f"   # the table's #70707c is too faint on the light tint


def _widget_dims_arg(base_w, base_h):
    def _dim(k, default):
        try:
            return max(80, min(int(request.args.get(k, default)), 800))
        except (TypeError, ValueError):
            return default
    w, h = _dim("w", base_w), _dim("h", base_h)
    zoom = w / base_w
    return {"w": w, "h": h, "base_w": base_w, "base_h": round(h / zoom, 2), "zoom": round(zoom, 5)}


def _widget_render_ctx():
    """Template context for one widget from the query string, or an error response.
    `kind=live` is the live-lido widget (medium only); default is the Fun+ Days widget."""
    family = (request.args.get("family") or "small").lower()
    appearance = "light" if request.args.get("appearance") == "light" else "dark"
    kind = (request.args.get("kind") or "forecast").lower()
    if kind == "live":
        if family != "medium":
            return None, (jsonify({"error": "the live widget is medium only"}), 400)
        spot, err = _live_spot_arg()
        if err:
            return None, err
        return {"kind": "live", "family": "medium", "appearance": appearance, "data": _live_payload(spot),
                **_widget_dims_arg(*_WIDGET_DIMS["medium"])}, None
    if family not in _WIDGET_DIMS:
        return None, (jsonify({"error": "family must be small, medium or large"}), 400)
    names, err = _widget_regions_arg()
    if err:
        return None, err
    names = names[:_WIDGET_N[family]]
    data = _widget_payload(names)
    regions = []
    for r in data["regions"]:
        c = r["colors"]
        if not c:
            tint, ink = ("#131316", "#e8e8f0") if appearance == "dark" else ("#ffffff", "#1e1e21")
        elif appearance == "dark":
            tint, ink = c["dark_bg"], (_FLAT_DARK_INK if r["category"] == "FLAT" else c["dark_text"])
        else:
            tint, ink = c["light_bg"], (_FLAT_LIGHT_INK if r["category"] == "FLAT" else c["light_text"])
        regions.append({**r, "tint": tint, "ink": ink})
    # w/h = the widget's point size on the phone (Scriptable sends it); the
    # mockup layout is zoomed to that box so the PNG is drawn 1:1, never
    # resampled by iOS. Default = the mockup's own px size.
    return {"kind": "forecast", "family": family, "appearance": appearance, "regions": regions,
            **_widget_dims_arg(*_WIDGET_DIMS[family]),
            "stamp": f"Last update {data['last_update']['label']}"}, None


def _widget_template(ctx: dict) -> str:
    return "widget_live_render.html" if ctx.get("kind") == "live" else "widget_render.html"


@app.route("/widget/render")
def widget_render():
    ctx, err = _widget_render_ctx()
    if err:
        return err
    return render_template(_widget_template(ctx), **ctx)


def _widget_scale_arg() -> int:
    try:
        return max(1, min(int(request.args.get("scale", 3)), 4))
    except (TypeError, ValueError):
        return 3


def _widget_full_png(ctx: dict, scale: int) -> Path | None:
    """Render (or reuse) the full widget PNG for a template context."""
    import hashlib, subprocess
    html = render_template(_widget_template(ctx), **ctx)
    key = hashlib.sha1(f"{scale}:{html}".encode()).hexdigest()
    _WIDGET_PNG_DIR.mkdir(parents=True, exist_ok=True)
    out = _WIDGET_PNG_DIR / f"{key}.png"
    vw, vh = ctx["w"], ctx["h"]
    with _widget_render_lock:
        if out.exists():
            return out
        with _tempfile.TemporaryDirectory() as td:
            src = Path(td) / "widget.html"
            src.write_text(html)
            tmp_png = Path(td) / "widget.png"
            cmd = [_CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                   f"--user-data-dir={td}/profile", f"--window-size={vw},{vh}",
                   f"--force-device-scale-factor={scale}", "--virtual-time-budget=6000",
                   f"--screenshot={tmp_png}", src.as_uri()]
            # Chrome writes the PNG early and then lingers ~40 s before
            # exiting; poll for the file and kill it (make_interface_guide.py
            # hit the same thing).
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.time() + 30
            while time.time() < deadline and not tmp_png.exists():
                time.sleep(0.2)
            time.sleep(0.3)
            proc.kill()
            proc.wait()
            if not tmp_png.exists():
                return None
            os.replace(tmp_png, out)
        # keep the cache small: anything older than a day is a stale model run
        cutoff = time.time() - 86400
        for f in _WIDGET_PNG_DIR.glob("*.png"):
            if f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                except OSError:
                    pass
    return out


def _widget_png_response(path: Path, ctx: dict | None = None):
    resp = send_from_directory(_WIDGET_PNG_DIR, path.name, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-cache"
    # where a tap on this widget should go: the live widget opens the Lido
    # Surfline page (the regional view's spot link), the Fun+ Days widget the site
    tap = (ctx or {}).get("data", {}).get("tap_url") if (ctx or {}).get("kind") == "live" else None
    resp.headers["X-Tap-Url"] = tap or _WIDGET_SITE
    return resp


@app.route("/widget/image.png")
def widget_image():
    ctx, err = _widget_render_ctx()
    if err:
        return err
    out = _widget_full_png(ctx, _widget_scale_arg())
    if out is None:
        return jsonify({"error": "render failed"}), 503
    return _widget_png_response(out, ctx)


def _png_decode(data: bytes):
    """Minimal PNG reader for Chrome's screenshots (8-bit RGB/RGBA, not
    interlaced) → (h, w, 3) uint8. Pillow isn't a dependency and sips drops
    a 0 crop offset, so the tile crop is done here."""
    import struct, zlib
    import numpy as np
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, hdr = 8, [], None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + n
    w, h, depth, ctype, _, _, interlace = hdr
    if depth != 8 or ctype not in (2, 6) or interlace:
        raise ValueError(f"unsupported PNG: depth {depth} type {ctype} interlace {interlace}")
    bpp = 4 if ctype == 6 else 3
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.int32)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = np.frombuffer(raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.int32)
        if f == 0:
            cur = line
        elif f == 1:                       # Sub: cumulative per channel
            cur = np.cumsum(line.reshape(-1, bpp), axis=0).reshape(-1) & 255
        elif f == 2:                       # Up
            cur = (line + prev) & 255
        else:                              # Average / Paeth: sequential in x
            cur = line.copy()
            c = cur.tolist(); p = prev.tolist()
            for x in range(stride):
                a = c[x - bpp] if x >= bpp else 0
                b = p[x]
                if f == 3:
                    c[x] = (c[x] + ((a + b) >> 1)) & 255
                else:
                    cc = p[x - bpp] if x >= bpp else 0
                    pa, pb, pc = abs(b - cc), abs(a - cc), abs(a + b - 2 * cc)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else cc)
                    c[x] = (c[x] + pred) & 255
            cur = np.array(c, np.int32)
        out[y] = cur
        prev = cur
    return out.reshape(h, w, bpp)[:, :, :3]


_widget_decoded: dict = {}      # full-PNG path → decoded array (one per render, tiny)


def _widget_tile_png(full: Path, x: int, y: int, w: int, h: int) -> bytes:
    with _widget_render_lock:
        arr = _widget_decoded.get(full)
        if arr is None:
            arr = _png_decode(full.read_bytes())
            _widget_decoded.clear()
            _widget_decoded[full] = arr
    return bathy._png(np.ascontiguousarray(arr[y:y + h, x:x + w]))


def _split_pts(total: int, n: int) -> list[int]:
    """Integer point widths summing to `total` — same rule as the script."""
    base = total // n
    return [base + (1 if i < total - base * n else 0) for i in range(n)]


@app.route("/widget/tile.png")
def widget_tile():
    """One tile of the widget (`cols`×`rows` grid, zero-based `col`/`row`).
    Scriptable recompresses any image it loads in a widget above ~500 k px,
    so the phone fetches tiles and never the full image."""
    ctx, err = _widget_render_ctx()
    if err:
        return err
    try:
        cols, rows = max(1, min(int(request.args["cols"]), 8)), max(1, min(int(request.args["rows"]), 8))
        col, row = int(request.args["col"]), int(request.args["row"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "cols, rows, col, row required"}), 400
    if not (0 <= col < cols and 0 <= row < rows):
        return jsonify({"error": "tile out of range"}), 400
    scale = _widget_scale_arg()
    full = _widget_full_png(ctx, scale)
    if full is None:
        return jsonify({"error": "render failed"}), 503
    out = _WIDGET_PNG_DIR / f"{full.stem}_{cols}x{rows}_{col}_{row}.png"
    if not out.exists():
        xs, ys = _split_pts(ctx["w"], cols), _split_pts(ctx["h"], rows)
        png = _widget_tile_png(full, sum(xs[:col]) * scale, sum(ys[:row]) * scale,
                               xs[col] * scale, ys[row] * scale)
        tmp = out.with_suffix(".tmp.png")
        tmp.write_bytes(png)
        os.replace(tmp, out)
    return _widget_png_response(out, ctx)


_WIDGET_DIR = Path(__file__).resolve().parent / "widget"


@app.route("/widget/<path:filename>")
def widget_file(filename: str):
    """The Scriptable widget script. A two-line stub on the phone fetches and
    evals this, so edits ship through git/autopull without touching the phone."""
    return send_from_directory(_WIDGET_DIR, filename, mimetype="application/javascript")


def _review_inline_config() -> str:
    """Shared inline config for /review and /seasons: spots, the live
    categorization scheme (for the fun+ definition sentence and colours),
    today's date and the ledger floor year."""
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _Zi
    from config import TIMEZONE as _TZ
    payload = {
        "spots":      SPOTS,
        "categories": swell_rules.CATEGORIES,
        "colors":     swell_rules.COLORS,
        "swell_bands": [{"period_ub": b["period_ub"], "rules": b["rules"]}
                        for b in swell_rules.load_bands()],   # fun+ definition text
        "wind_rating": wind_rules.load_config(),                 # surfable-wind text
        "today":      _dt.now(_Zi(_TZ)).date().isoformat(),
        "first_year": 2019,   # floor of the custom season/year picker (ledgers + wind archive reach 2019)
    }
    return _json.dumps(payload, separators=(",", ":"))


@app.route("/review")
def review_page():
    """Conditions Reviewer — per-region histograms of observed swell ratings,
    daily peak energy and primary period over a chosen window (fun_days.py)."""
    return render_template("review.html", inline_config=_review_inline_config())


@app.route("/seasons")
def seasons_page():
    """Seasonal Analysis — per-region season-by-year tables of observed day
    counts and drought lengths (fun_days.season_tables)."""
    return render_template("seasons.html", inline_config=_review_inline_config())


@app.route("/api/review")
def api_review():
    """Ledger rows per buoy for [start, end] (ISO dates, end clamped to
    today). Missing ledger years are built on demand from the obs archive."""
    from datetime import date as _date
    try:
        start = _date.fromisoformat(request.args.get("start", ""))
        end   = _date.fromisoformat(request.args.get("end", ""))
    except ValueError:
        return jsonify({"error": "start/end must be YYYY-MM-DD"}), 400
    if end < start or start < _date(2000, 1, 1):
        return jsonify({"error": "bad range"}), 400
    data = _review_payload(start.isoformat(), end.isoformat())
    if data is None:
        return jsonify({"error": "data unavailable"}), 503
    return jsonify(data)


@app.route("/api/review/seasons")
def api_review_seasons():
    """Season-by-year fun+/flat/solid-or-firing day counts and drought
    lengths per buoy, every ledger year on disk from 2019
    (fun_days.season_tables, 1 h TTL). Feeds /seasons."""
    data = _season_tables()
    if data is None:
        return jsonify({"error": "data unavailable"}), 503
    return jsonify(data)


@app.route("/")
def index():
    # Inline /api/config data into the HTML to save one round-trip on initial load.
    return render_template("index.html",
                           inline_config=_json.dumps(_config_payload(), separators=(',', ':')))

@app.route("/csc")
def csc_page():
    """CSC2 evaluation page: archive coverage, model registry, metric tables."""
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


# ─── G-Land (Grajagan, East Java) ────────────────────────────────────────────
# Standalone from the main dashboard on purpose: there is no NDBC buoy and no
# CO-OPS tide station within thousands of km of G-Land, so none of the
# regions.yaml plumbing applies. See gland.py for the source list.

@app.route("/gland")
def gland_page():
    """G-Land forecast page — cheat sheet + live model data for Grajagan."""
    import gland as _gland
    payload = {
        "lat": _gland.GLAND_LAT,
        "lon": _gland.GLAND_LON,
        "tz": _gland.GLAND_TZ,
        "surfline_url": _gland.SURFLINE_URL,
        "sections": _gland.SECTIONS,
        "reef_line": _gland.REEF_LINE,
        "harbour_channel": _gland.HARBOUR_CHANNEL,
        "point_tip": _gland.POINT_TIP,
        "upstream": _gland.UPSTREAM_BUOYS,
        "window_core": _gland.WINDOW_CORE,
        "window_edge": _gland.WINDOW_EDGE,
        "swell_bands": _gland.SWELL_BANDS,
        "swell_node": {"lat": _gland.SWELL_NODE_LAT,
                       "lon": _gland.SWELL_NODE_LON,
                       "km": _gland.SWELL_NODE_KM},
        # G-Land's own ladder, colours and match rules — DREAMY and BIG do
        # not exist site-wide, so these must not come from swell_rules.
        "categories": _gland.GLAND_CATEGORIES,
        "colors": _gland.GLAND_COLORS,
        "rules": _gland.gland_rules_payload(),
    }
    return render_template(
        "gland.html",
        inline_config=_json.dumps(payload, separators=(',', ':')),
    )


@app.route("/api/gland/all")
def api_gland_all():
    """Everything the G-Land page needs in one call: both wave models, tide,
    wind, the Western Australian sentinel buoys, and the moon phase."""
    import gland as _gland
    data = _gland.fetch_all()
    meta = data.get("meta") or {}
    if not meta.get("have_gfs") and not meta.get("have_euro"):
        return jsonify({"error": "no wave model data available"}), 503
    return jsonify(data)


@app.route("/gland/tuner")
def gland_tuner_page():
    """G-Land-only swell category tuner. Writes gland-swell-categorization.toml
    and touches nothing the rest of the site reads — the site-wide scheme in
    swell-categorization-scheme.toml is edited at /tuner and is unaffected."""
    import gland as _gland
    payload = {
        "rules": _gland.gland_rules_payload(),
        "categories": _gland.GLAND_CATEGORIES,
        "colors": _gland.GLAND_COLORS,
    }
    return render_template(
        "gland-tuner.html",
        inline_config=_json.dumps(payload, separators=(',', ':')),
    )


@app.route("/api/gland/tuner/save", methods=["POST"])
def api_gland_tuner_save():
    """Persist G-Land rule edits and reload just the G-Land rules."""
    import gland as _gland
    payload = request.get_json(silent=True) or {}
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        return jsonify({"error": "no rules supplied"}), 400

    def pair(v):
        lo, hi = v
        fmt = lambda x: '"inf"' if str(x).lower() == "inf" else repr(float(x))
        return f"[{fmt(lo)}, {fmt(hi)}]"

    lines = [
        "# G-Land Swell Categorization Scheme",
        "# G-LAND ONLY — auto-written by /api/gland/tuner/save; edit at /gland/tuner.",
        "# Ordered rules: the FIRST match wins, so order is priority.",
        "#   period/height/direction are [min, max]; \"inf\" is unbounded;",
        "#   min is inclusive, max exclusive. Omit direction to ignore it.",
        "",
    ]
    for r in rules:
        lines.append("[[rule]]")
        lines.append(f'category  = "{r["category"]}"')
        lines.append(f'period    = {pair(r.get("period", [0, "inf"]))}')
        lines.append(f'height    = {pair(r.get("height", [0, "inf"]))}')
        if r.get("direction"):
            lines.append(f'direction = {pair(r["direction"])}')
        lines.append("")
    try:
        _atomic_write_text(_gland.GLAND_TOML, "\n".join(lines) + "\n")
        _gland.reload_gland_rules()
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"status": "saved", "rules": len(rules)})


@app.route("/api/gland/tides")
def api_gland_tides():
    """Tide highs/lows for a date range (lookup tool). Capped at 14 days."""
    import gland as _gland
    import datetime as _dt
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    try:
        d0 = _dt.date.fromisoformat(start)
        d1 = _dt.date.fromisoformat(end)
    except ValueError:
        return jsonify({"error": "start and end must be YYYY-MM-DD"}), 400
    if d1 < d0:
        d0, d1 = d1, d0
    if (d1 - d0).days > 13:
        d1 = d0 + _dt.timedelta(days=13)
    data = _gland.fetch_gland_tide_range(d0.isoformat(), d1.isoformat())
    if not data:
        return jsonify({"error": "tide data unavailable for that range"}), 503
    return jsonify(data)


@app.route("/api/gland/summary")
def api_gland_summary():
    """Fun+ Days for G-Land, for the main dashboard's overview row."""
    import gland as _gland
    data = _gland.fun_plus_summary()
    if not data:
        return jsonify({"error": "summary unavailable"}), 503
    return jsonify(data)


@app.route("/api/gland/history")
def api_gland_history():
    """Past 14 days at G-Land. GFS + wind come from Open-Meteo's own past
    analysis and tide from the harmonic fit; EURO is the only locally
    archived piece (gland_euro_archive.py)."""
    import gland as _gland
    try:
        days = int(request.args.get("days", _gland.HISTORY_DAYS))
    except ValueError:
        days = _gland.HISTORY_DAYS
    days = max(1, min(days, 30))
    data = _gland.fetch_gland_history(days)
    if not data:
        return jsonify({"error": "history unavailable"}), 503
    return jsonify(data)


@app.route("/api/gland/upstream")
def api_gland_upstream():
    """Just the WA sentinel buoys — used for the standalone refresh."""
    import gland as _gland
    data = _gland.fetch_upstream_buoys()
    if data is None:
        return jsonify({"error": "AODN unavailable"}), 503
    return jsonify(data)


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

    The EURO cycle is derived from the fetched series (csc2.logger.euro_cycle_id)
    and only changes twice a day, so response-level caching is safe. The cache warmer pre-fills this for
    every east buoy on startup, so users essentially never pay the cold
    cost (~7 s for CMEMS + ~600 ms for the 3 model predictions)."""
    from csc2.registry import selection_payload
    from csc2.predict import predict_for_cycle
    from csc2.schema import buoy_meta, CSC2_MODELS_DIR
    from csc2.logger import euro_cycle_from_records
    from cache import age_of
    from datetime import datetime, timedelta, timezone as _tz
    from waves_cmems import fetch_cmems_point

    try:
        meta = buoy_meta(buoy_id)
    except KeyError:
        return None

    now = datetime.now(_tz.utc)

    try:
        euro_recs = fetch_cmems_point(meta["lat"], meta["lon"]) or []
        euro_err = None
    except Exception as e:
        euro_recs, euro_err = [], f"{type(e).__name__}: {e}"
    # Lead hours are measured from the EURO run (the quantity being corrected);
    # GFS at the same wall clock may be one run fresher, as in training. The
    # series came from the TTL cache, so judge the run by when it was fetched.
    age = age_of(fetch_cmems_point, meta["lat"], meta["lon"])
    fetched_at = now - timedelta(seconds=age) if age else now
    cycle_utc = euro_cycle_from_records(fetched_at, euro_recs) or now.replace(
        hour=0 if now.hour < 12 else 12, minute=0, second=0, microsecond=0).strftime("%Y%m%dT%HZ")
    try:
        gfs_recs = fetch_wave_forecast(meta["lat"], meta["lon"], "GFS") or []
        gfs_err = None
    except Exception as e:
        gfs_recs, gfs_err = [], f"{type(e).__name__}: {e}"

    # predict_for_cycle inner-joins EURO ∩ GFS, so if either feed is empty the
    # payload is guaranteed useless. Return None (skip_none → NOT cached) so
    # the next request retries instead of pinning the failure for 30 min.
    if not euro_recs or not gfs_recs:
        print(f"[csc2] forecast {buoy_id}: euro={len(euro_recs)} gfs={len(gfs_recs)} "
              f"rows (euro_err={euro_err}, gfs_err={gfs_err}) — not caching")
        return None

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
        return jsonify({"error": "forecast inputs unavailable, retry later"}), 503
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

# PNG favicon fallbacks — iOS Safari does not render SVG favicons in tabs/bookmarks.
@app.route("/favicon-16.png")
def favicon_16():
    return send_from_directory(app.root_path, "favicon-16.png", mimetype="image/png")

@app.route("/favicon-32.png")
def favicon_32():
    return send_from_directory(app.root_path, "favicon-32.png", mimetype="image/png")

@app.route("/favicon-192.png")
def favicon_192():
    return send_from_directory(app.root_path, "favicon-192.png", mimetype="image/png")

@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return send_from_directory(app.root_path, "apple-touch-icon.png", mimetype="image/png")

# iOS probes this exact path (and older iOS prefers it) when no <link> matches.
@app.route("/apple-touch-icon-precomposed.png")
def apple_touch_icon_precomposed():
    return send_from_directory(app.root_path, "apple-touch-icon-precomposed.png", mimetype="image/png")


# ─── Cache-Control headers ──────────────────────────────────────────────────
# Freshness-first policy (2026-07): browser max-age + edge stale-while-
# revalidate meant a new model run took 2-3 reloads to appear (first reload
# served stale from the edge while revalidating in background; the browser's
# max-age then re-served that same stale copy on the next reload). API
# responses are now no-store — never cached by browser or Cloudflare edge —
# so every open reaches the origin, where the TTL cache + 30-min warmer keep
# responses fast. Freshness is bounded by cache.py TTLs alone. HTML is
# no-cache so autopulled UI changes appear on the next open.
@app.after_request
def _add_cache_headers(response):
    if request.method != "GET":
        return response
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif request.path.startswith("/widget/"):
        response.headers["Cache-Control"] = "no-cache"
    elif request.path == "/favicon.svg" or request.path.startswith(("/favicon-", "/apple-touch-icon")):
        # Icons: always revalidate against origin (ETag) so an icon swap can
        # never get stuck in the Cloudflare edge / browser cache for hours.
        response.headers["Cache-Control"] = "no-cache"
    elif response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-cache"
    return response


# ─── Background cache warming ─────────────────────────────────────────────────
_WARM_INTERVAL = 1800   # 30 minutes — well within TTL of 3600s


def _warm_all_caches():
    """Pre-fetch all data so user requests always hit warm cache.

    Independent sources warm concurrently; the six api.open-meteo.com wind
    calls stay sequential within their group to keep today's request pattern
    (no concurrent-429 risk). CSC2 runs after the wave groups because its
    per-buoy fetches reuse the TTL-cache entries those groups populate."""
    t0 = time.monotonic()
    errors = []

    def _run(label, fn):
        try:
            fn()
        except Exception as e:
            errors.append(f"{label}: {e}")

    def _warm_open_meteo_wind():
        for model in ("EURO", "GFS"):
            _run(f"wind_grid/{model}", lambda m=model: fetch_wind_grid(m))
            _run(f"wind_forecast/{model}", lambda m=model: fetch_wind_forecast_grid(m))
        for model in ("EURO", "GFS"):
            _run(f"region_wind/{model}", lambda m=model: fetch_region_wind_forecasts(m))

    def _warm_buoys():
        futures = {_buoy_pool.submit(fetch_buoy, s["buoy_id"]): s["name"] for s in SPOTS}
        for f in as_completed(futures, timeout=30):
            f.result()

    groups = [
        # GFS waves: batched through Open-Meteo (1 API call, all spots).
        ("wave/GFS", lambda: fetch_all_wave_forecasts("GFS")),
        # EURO: CMEMS, parallel across region buoys (~11 s for 7 buoys in testing).
        ("cmems",    fetch_all_cmems_wave_forecasts),
        ("wind",     _warm_open_meteo_wind),
        ("buoys",    _warm_buoys),
        ("tides",    fetch_tide_predictions),
    ]
    # Not _buoy_pool: _warm_buoys submits nested futures there; sharing one
    # pool risks starvation.
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        for f in [pool.submit(_run, label, fn) for label, fn in groups]:
            f.result()

    # CSC2 forecast — pre-compute predictions for every east buoy so the
    # /csc page never pays the cold cost. Skipped if no models are trained.
    def _warm_csc2():
        from csc2.schema import buoys_in as _csc2_buoys_in
        from csc2.registry import list_models as _csc2_list_models
        if _csc2_list_models("east"):
            for _bid in _csc2_buoys_in("east"):
                _run(f"csc2_forecast/{_bid}", lambda b=_bid: _csc2_forecast_payload(b, "east"))
    _run("csc2_forecast", _warm_csc2)
    # Observed fun+ ledger — after the wind group so today's gate hours come
    # from the freshly warmed EURO region-wind entry.
    _run("fun_days", _fun_days_all)

    elapsed = time.monotonic() - t0
    if errors:
        print(f"[cache-warm] done in {elapsed:.1f}s with {len(errors)} errors: "
              + "; ".join(errors))
    else:
        print(f"[cache-warm] all caches refreshed in {elapsed:.1f}s")


def _cache_warmer_loop():
    """Background thread: warm caches on startup and then every WARM_INTERVAL seconds."""
    # Initial warm on startup (brief yield so Waitress binds first)
    time.sleep(0.5)
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
    print("  ─────────────────────────────────")
    print(f"  Host    → {HOST}:{PORT}")
    if HOST == "0.0.0.0":
        print(f"  Local   → http://127.0.0.1:{PORT}")
        print(f"  Network → http://{_local_ip}:{PORT}")
    else:
        print(f"  Local   → http://{HOST}:{PORT}")
    print(f"  Debug   → {'on' if DEBUG_MODE else 'off'}")
    print(f"  Warmer  → every {_WARM_INTERVAL}s")
    print("  Press Ctrl+C to stop.\n")

    # Start background cache warmer
    _warmer = threading.Thread(target=_cache_warmer_loop, daemon=True)
    _warmer.start()
    bathy.prewarm_async()   # default-view basemap tiles; no-op once rendered

    serve(app, host=HOST, port=PORT, threads=8)
