"""
colesurfs — Wind Data Fetcher
  • fetch_spot_wind()                       → per-spot current wind (Open-Meteo)
  • fetch_spot_wind_forecasts()             → per-spot hourly wind for WIND table row (Open-Meteo)
  • fetch_region_wind_forecasts(model_key)  → hourly wind + gust for all WIND_SPOTS
                                              (table cells, ratings, Fun+ gate, widgets)
  • estimate_model_run(model_key)           → best guess of which model run is current

Spot winds are the MAP'S OWN FIELD: fetch_region_wind_forecasts samples the
wind_field series (the model's 0.25° GRIB u/v/gust the map paints) at each
spot with the page's interpolation, so a table cell, its map label and the
colour under the dot are one number. Open-Meteo point forecasts remain only
for history hours older than the GRIB store (the −240 h strip) and as a
stand-in before the first run lands — always for the SAME model
(WIND_MODELS: EURO → ecmwf_ifs, GFS → gfs_seamless), never another model's
wind under this model's name.
"""
import threading
import time
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from cache import ttl_cache, record_api_calls
from config import (
    TIMEZONE, FORECAST_DAYS, WIND_MODELS, MODEL_UPDATE_HOURS_UTC,
    ms_to_kts, ms_to_mph, degrees_to_cardinal, SPOTS, WIND_SPOTS,
)
import wind_field

FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Negative cache: when a request fails, don't retry for this many seconds.
_NEGATIVE_CACHE_SEC = 1800  # 30 min cooldown after API failure (e.g. 429 rate limit)

_negative_cache: dict[str, float] = {}   # key → monotonic time of failure


def _is_negative_cached(key: str) -> bool:
    ts = _negative_cache.get(key)
    if ts is None:
        return False
    if time.monotonic() - ts < _NEGATIVE_CACHE_SEC:
        return True
    del _negative_cache[key]
    return False


def _set_negative_cache(key: str):
    _negative_cache[key] = time.monotonic()


# ─── Model run estimation ────────────────────────────────────────────────────

def estimate_model_run(model_key: str = "EURO") -> dict:
    """
    Estimate which model run Open-Meteo is currently serving.
    Returns {run_utc: "00Z", run_time: "2026-04-01T00:00Z", available_since: "..."}.
    """
    now = datetime.now(timezone.utc)
    update_hours = MODEL_UPDATE_HOURS_UTC.get(model_key, [7, 19])

    # Walk backwards through update hours to find the most recent one
    for days_back in range(2):
        check_day = now - timedelta(days=days_back)
        for h in sorted(update_hours, reverse=True):
            available_at = check_day.replace(hour=h, minute=0, second=0, microsecond=0)
            if available_at <= now:
                # This update hour is in the past — this is the current run.
                # GFS: available ~4h after init. EURO (CMEMS): the 00Z run
                # lands ~08:30Z and the 12Z run ~20:50Z, so 10Z → 00Z, 21Z → 12Z.
                if model_key == "GFS":
                    init_time = available_at - timedelta(hours=4)
                else:
                    init_time = available_at.replace(hour=0 if available_at.hour < 12 else 12)
                run_label = f"{init_time.hour:02d}Z"
                run_date  = init_time.strftime("%Y-%m-%d")

                # Find the next update hour after now
                next_available = None
                for fd in range(3):
                    future_day = now + timedelta(days=fd)
                    for fh in sorted(update_hours):
                        candidate = future_day.replace(hour=fh, minute=0, second=0, microsecond=0)
                        if candidate > now:
                            next_available = candidate
                            break
                    if next_available:
                        break
                hours_to_next = None
                if next_available:
                    hours_to_next = round((next_available - now).total_seconds() / 3600, 1)

                return {
                    "run_utc":         run_label,
                    "run_date":        run_date,
                    "run_time":        init_time.strftime("%Y-%m-%dT%H:%MZ"),
                    "available_since": available_at.strftime("%Y-%m-%dT%H:%MZ"),
                    "hours_to_next":   hours_to_next,
                    "model":           model_key,
                }

    return {"run_utc": "??Z", "run_date": None, "run_time": None,
            "available_since": None, "hours_to_next": None, "model": model_key}


def _new_run_available_since(model_key: str, cache_age_sec: float,
                             hours_map: dict | None = None) -> bool:
    """
    Check if a new model run has likely become available since the cache was populated.
    Returns True if we should re-fetch, False if cached data is still the latest.
    `hours_map` selects the publication schedule (default: wave-model hours).
    """
    if cache_age_sec is None:
        return True  # no cache → must fetch

    now = datetime.now(timezone.utc)
    cached_at = now - timedelta(seconds=cache_age_sec)
    update_hours = (hours_map or MODEL_UPDATE_HOURS_UTC).get(model_key, [7, 19])

    # Check if any update hour falls between cached_at and now
    for days_back in range(2):
        check_day = now - timedelta(days=days_back)
        for h in update_hours:
            update_time = check_day.replace(hour=h, minute=0, second=0, microsecond=0)
            if cached_at < update_time <= now:
                return True

    return False


def make_new_run_checker(hours_map: dict):
    """Checker bound to a specific publication schedule (wave vs wind hours)
    for cache.model_aware_cache — waves_cmems binds the CMEMS hours."""
    return lambda model_key, age: _new_run_available_since(model_key, age, hours_map)


# ─── Per-spot current wind ────────────────────────────────────────────────────

def _current_to_spot_wind(cur: dict) -> dict:
    spd  = cur.get("wind_speed_10m")
    dirn = cur.get("wind_direction_10m")
    gust = cur.get("wind_gusts_10m")
    return {
        "speed_ms":      spd,
        "direction_deg": dirn,
        "gust_ms":       gust,
        "speed_kts":     ms_to_kts(spd),
        "gust_kts":      ms_to_kts(gust),
    }


@ttl_cache(ttl_seconds=3600, skip_none=True)
def fetch_all_spot_winds() -> dict | None:
    """Current wind for ALL SPOTS in one multi-location call (1 request
    instead of N). Returns {spot_name: spot_wind_dict} or None on failure;
    callers fall back to per-spot fetch_spot_wind."""
    params = {
        "latitude":  ",".join(str(s["lat"]) for s in SPOTS),
        "longitude": ",".join(str(s["lon"]) for s in SPOTS),
        "current":   "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms",
        "timezone":  TIMEZONE,
    }
    try:
        record_api_calls("spot_wind_batch", len(SPOTS))
        r = requests.get(FORECAST_API, params=params, timeout=15,
                         headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        print("[spot_wind_batch] timeout")
        return None
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        print(f"[spot_wind_batch] HTTP {code}")
        return None
    except Exception as e:
        print(f"[spot_wind_batch] {type(e).__name__}: {e}")
        return None

    if isinstance(data, dict):
        data = [data]
    if not data or not isinstance(data, list) or data[0].get("error"):
        return None

    result = {}
    for i, spot in enumerate(SPOTS):
        cur = data[i].get("current", {}) if i < len(data) else {}
        result[spot["name"]] = _current_to_spot_wind(cur) if cur else None
    return result if any(v is not None for v in result.values()) else None


@ttl_cache(ttl_seconds=3600, skip_none=True)
def fetch_spot_wind(lat: float, lon: float) -> dict | None:
    params = {
        "latitude": lat, "longitude": lon,
        "current":  "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms",
        "timezone": TIMEZONE,
    }
    try:
        record_api_calls("spot_wind", 1)
        r = requests.get(FORECAST_API, params=params, timeout=12,
                         headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
        d = r.json()
    except requests.exceptions.Timeout:
        print(f"[spot_wind] ({lat},{lon}) timeout")
        return None
    except Exception as e:
        print(f"[spot_wind] ({lat},{lon}) {type(e).__name__}: {e}")
        return None
    return _current_to_spot_wind(d.get("current", {}))


# ─── Per-spot hourly wind forecast (for WIND table row) ───────────────────────
@ttl_cache(ttl_seconds=3600, skip_none=True)
def fetch_spot_wind_forecasts() -> dict | None:
    """
    Hourly wind forecast for all configured SPOTS via a single multi-location request.
    Returns {spot_name: [{time, speed_kts, direction_deg, gust_kts}, ...]}
    """
    lats = ",".join(str(s["lat"]) for s in SPOTS)
    lons = ",".join(str(s["lon"]) for s in SPOTS)

    params = {
        "latitude":        lats,
        "longitude":       lons,
        "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms",
        "forecast_days":   FORECAST_DAYS,
        "timezone":        TIMEZONE,
    }

    try:
        record_api_calls("spot_wind_forecasts", len(SPOTS))
        r = requests.get(FORECAST_API, params=params, timeout=25,
                         headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[wind] spot forecasts: {e}")
        return None

    if isinstance(data, dict):
        data = [data]
    if not data or not isinstance(data, list):
        return None

    result = {}
    for i, spot in enumerate(SPOTS):
        if i >= len(data):
            break
        h      = data[i].get("hourly", {})
        times  = h.get("time",               [])
        speeds = h.get("wind_speed_10m",     [])
        dirs   = h.get("wind_direction_10m", [])
        gusts  = h.get("wind_gusts_10m",     [])

        records = []
        for j, t in enumerate(times):
            spd  = speeds[j] if j < len(speeds) else None
            dirn = dirs[j]   if j < len(dirs)   else None
            gust = gusts[j]  if j < len(gusts)  else None
            records.append({
                "time":          t,
                "speed_kts":     ms_to_kts(spd),
                "direction_deg": dirn,
                "gust_kts":      ms_to_kts(gust),
            })
        result[spot["name"]] = records

    return result


# ─── Regional wind spot hourly forecasts (table cells, ratings, Fun+ gate) ────

def _unique_wind_coords():
    """WIND_SPOTS deduplicated by (lat, lon) — spots shared between regions
    are one point. Returns (coords, index per WIND_SPOTS entry)."""
    coords, idx, spot_to = [], {}, []
    for ws in WIND_SPOTS:
        key = (ws["lat"], ws["lon"])
        if key not in idx:
            idx[key] = len(coords)
            coords.append(key)
        spot_to.append(idx[key])
    return coords, spot_to


def _record(t: str, spd_ms, dir_deg, gust_ms) -> dict:
    d = None if dir_deg is None else int(round(float(dir_deg))) % 360
    return {
        "time":               t,
        "speed_mph":          ms_to_mph(spd_ms),
        "direction_deg":      d,
        "direction_cardinal": degrees_to_cardinal(d),
        "gust_mph":           ms_to_mph(gust_ms),
        "gust_cardinal":      degrees_to_cardinal(d),
    }


@ttl_cache(ttl_seconds=3600, skip_none=True)
def _open_meteo_region_hours(model_key: str, past_days: int) -> dict | None:
    """Open-Meteo's hourly point wind for WIND_MODELS[model_key] at every
    unique spot coordinate: {unique index: [record…]}. Used only where the
    GRIB series has nothing (history older than three days; everything before
    the first run lands). The requested model or nothing — a failure here
    must never put another model's wind under this model's name."""
    coords, _ = _unique_wind_coords()
    if not coords:
        return {}
    neg_key = f"region_wind:{model_key}"
    if _is_negative_cached(neg_key):
        print("[region_wind] skipping Open-Meteo — negative cached (rate limited recently)")
        return None
    model_id = WIND_MODELS.get(model_key)
    params = {
        "latitude":        ",".join(str(c[0]) for c in coords),
        "longitude":       ",".join(str(c[1]) for c in coords),
        "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms",
        "forecast_days":   FORECAST_DAYS,
        "timezone":        TIMEZONE,
    }
    if model_id:
        params["models"] = model_id
    if past_days > 0:
        params["past_days"] = past_days
    print(f"[region_wind] Open-Meteo {model_id or 'default'}: {len(coords)} pts, past_days={past_days}…")
    try:
        record_api_calls("region_wind", len(coords))
        r = requests.get(FORECAST_API, params=params, timeout=30,
                         headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        print(f"[region_wind] Open-Meteo ({model_id or 'default'}): {e}")
        if "429" in str(e):
            _set_negative_cache(neg_key)
        return None
    if isinstance(raw, dict):
        raw = [raw]
    if not raw or not isinstance(raw, list) or raw[0].get("error"):
        print(f"[region_wind] Open-Meteo API error ({model_id or 'default'}): "
              f"{raw[0].get('reason', '?') if raw and isinstance(raw, list) else raw}")
        return None
    out = {}
    for i in range(len(coords)):
        h = raw[i].get("hourly", {}) if i < len(raw) else {}
        times, speeds = h.get("time", []), h.get("wind_speed_10m", [])
        dirs, gusts = h.get("wind_direction_10m", []), h.get("wind_gusts_10m", [])
        out[i] = [_record(t, speeds[j] if j < len(speeds) else None,
                          dirs[j] if j < len(dirs) else None,
                          gusts[j] if j < len(gusts) else None)
                  for j, t in enumerate(times)]
    return out


_region_memo: dict[tuple, dict] = {}
_region_lock = threading.Lock()


def fetch_region_wind_forecasts(model_key: str = "EURO", past_days: int = 0) -> dict | None:
    """
    Hourly wind + gust for all WIND_SPOTS, local time, respecting model_key:
    {spot_name: [{time, speed_mph, direction_deg, direction_cardinal,
                  gust_mph, gust_cardinal}, ...]}

    Every hour the map's series covers (three days back to the newest run's
    +240 h) is the map's own field — wind_field.sample(): the model's 0.25°
    u/v/gust, linear between steps, bilinear in the grid, read at the spot —
    so the table cell, the spot label and the colour beneath the dot agree.
    Hours before the series (the history strip, `past_days` up to 30) and,
    before the first run lands, the whole window come from Open-Meteo's point
    forecast for the same model. Hours with no source are left out.

    Memoised per (model, past_days) on the series build (a new run, a top-up
    such as gusts landing, or the hourly window move rebuild the series and
    so this); building is local arithmetic.
    """
    if not WIND_SPOTS:
        return {}
    past_days = max(0, min(int(past_days or 0), 30))
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=past_days)
    s = wind_field.series(model_key)
    key = (model_key, past_days, s["meta"]["built"] if s else None, now.strftime("%Y%m%d%H"))
    with _region_lock:
        hit = _region_memo.get(key)
    if hit is not None:
        return hit

    coords, spot_to = _unique_wind_coords()
    end = (datetime.fromtimestamp(int(s["ms"][-1]) / 1000, timezone.utc).astimezone(tz) if s
           else start + timedelta(days=FORECAST_DAYS))
    n_hours = int((end - start).total_seconds() // 3600) + 1
    times = [start + timedelta(hours=k) for k in range(max(n_hours, 0))]
    labels = [t.strftime("%Y-%m-%dT%H:%M") for t in times]
    ms = [int(t.timestamp() * 1000) for t in times]

    per_point: list[dict[str, dict]] = [{} for _ in coords]
    sampled = wind_field.sample(model_key, coords, ms) if s else None
    if sampled:
        spd, dirn, gust = sampled["speed_ms"], sampled["dir_deg"], sampled["gust_ms"]
        for k, t in enumerate(labels):
            for i in range(len(coords)):
                v = spd[k, i]
                if np.isnan(v):
                    continue
                g = gust[k, i]
                per_point[i][t] = _record(t, float(v), float(dirn[k, i]), None if np.isnan(g) else float(g))

    # Open-Meteo only for hours the series can't give: before its first
    # step (history) or all of them (no run yet).
    need = (s is None) or (s["ms"][0] > ms[0] if ms else False)
    if need:
        om = _open_meteo_region_hours(model_key, past_days)
        for i, recs in (om or {}).items():
            have = per_point[i]
            for r in recs:
                if r["time"] not in have and r["speed_mph"] is not None and r["time"] >= labels[0]:
                    have[r["time"]] = r

    result = {}
    for j, ws in enumerate(WIND_SPOTS):
        recs = per_point[spot_to[j]]
        result[ws["name"]] = [recs[t] for t in sorted(recs)]
    if not any(result.values()):
        return None
    with _region_lock:
        for k in [k for k in _region_memo if k[:2] == key[:2]]:
            _region_memo.pop(k, None)
        _region_memo[key] = result
    return result
