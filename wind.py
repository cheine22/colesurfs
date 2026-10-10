"""
colesurfs — Wind Data Fetcher
  • fetch_spot_wind()                       → per-spot current wind for table
  • fetch_spot_wind_forecasts()             → per-spot hourly wind for WIND table row
  • fetch_region_wind_forecasts(model_key)  → hourly wind for all WIND_SPOTS (regional mode)
  • estimate_model_run(model_key)           → best guess of which model run is current

Wind model is matched to the active wave model:
  EURO → ecmwf_ifs atmospheric model
  GFS  → gfs atmospheric model

The map's gridded wind field is NOT fetched here — see wind_field.py (0.25°
GRIB straight from NOMADS / ECMWF open data).
"""
import time
import requests
from datetime import datetime, timezone, timedelta
from cache import ttl_cache, record_api_calls
from config import (
    TIMEZONE, FORECAST_DAYS, WIND_MODELS, MODEL_UPDATE_HOURS_UTC,
    ms_to_kts, ms_to_mph, degrees_to_cardinal, SPOTS, WIND_SPOTS,
)

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


# ─── Regional wind spot hourly forecasts (for Regional Mode table) ─────────────
@ttl_cache(ttl_seconds=3600)
def fetch_region_wind_forecasts(model_key: str = "EURO", past_days: int = 0) -> dict | None:
    """
    Hourly wind + gust forecast for all WIND_SPOTS, respecting model_key.
    Returns {spot_name: [{time, speed_mph, direction_deg, direction_cardinal,
                          gust_mph, gust_cardinal}, ...]}
    Uses WIND_MODELS[model_key] atmospheric model (same as wind grid).
    Falls back to API default if the requested model fails.

    `past_days` (0..30) instructs Open-Meteo to include this many days of
    historical hours BEFORE today in the response. Used by the dashboard's
    historical-data toggle so the per-spot wind strip can show observed
    wind from the same model for each historical cell.

    Deduplicates spots that share the same lat/lon (e.g. spots appearing in
    multiple regions) so the API call uses only unique coordinates, saving
    quota and avoiding 429 rate-limit errors.
    """
    if not WIND_SPOTS:
        return {}

    neg_key = f"region_wind:{model_key}"
    if _is_negative_cached(neg_key):
        print("[region_wind] skipping — negative cached (rate limited recently)")
        return None

    # ── Deduplicate by lat/lon ──────────────────────────────────────────────
    # Build a list of unique (lat, lon) pairs and track which spot names
    # map to each unique location.
    unique_coords = []          # [(lat, lon), ...]
    coord_to_idx: dict[tuple, int] = {}   # (lat, lon) → index in unique_coords
    spot_to_unique: list[int] = []        # WIND_SPOTS index → unique_coords index

    for s in WIND_SPOTS:
        key = (s["lat"], s["lon"])
        if key not in coord_to_idx:
            coord_to_idx[key] = len(unique_coords)
            unique_coords.append(key)
        spot_to_unique.append(coord_to_idx[key])

    n_unique = len(unique_coords)
    model_id = WIND_MODELS.get(model_key)
    lats = ",".join(str(c[0]) for c in unique_coords)
    lons = ",".join(str(c[1]) for c in unique_coords)

    past_days = max(0, min(int(past_days or 0), 30))
    base_params = {
        "latitude":        lats,
        "longitude":       lons,
        "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms",
        "forecast_days":   FORECAST_DAYS,
        "timezone":        TIMEZONE,
    }
    if past_days > 0:
        base_params["past_days"] = past_days

    print(f"[region_wind] fetching {n_unique} unique pts "
          f"(from {len(WIND_SPOTS)} total spots)…")

    attempts = ([{**base_params, "models": model_id}] if model_id else []) + [base_params]
    data = None
    for params in attempts:
        try:
            record_api_calls("region_wind", n_unique)
            r = requests.get(FORECAST_API, params=params, timeout=30,
                             headers={"User-Agent": "ColeSurfs/1.0"})
            r.raise_for_status()
            raw = r.json()
        except Exception as e:
            print(f"[region_wind] fetch ({params.get('models', 'default')}): {e}")
            if "429" in str(e):
                _set_negative_cache(neg_key)
                break   # don't retry fallback model — also rate limited
            continue

        if isinstance(raw, dict):
            raw = [raw]
        if not raw or not isinstance(raw, list):
            continue
        if raw[0].get("error"):
            print(f"[region_wind] API error ({params.get('models', 'default')}): "
                  f"{raw[0].get('reason', '?')} — trying without model…")
            continue

        times = raw[0].get("hourly", {}).get("time", [])
        if times:
            data = raw
            break

    if not data:
        return None

    # ── Parse unique responses ─────────────────────────────────────────────
    unique_records: list[list[dict]] = []
    for i in range(n_unique):
        if i >= len(data):
            unique_records.append([])
            continue
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
                "time":               t,
                "speed_mph":          ms_to_mph(spd),
                "direction_deg":      dirn,
                "direction_cardinal": degrees_to_cardinal(dirn),
                "gust_mph":           ms_to_mph(gust),
                "gust_cardinal":      degrees_to_cardinal(dirn),
            })
        unique_records.append(records)

    # ── Map unique results back to all spot names ──────────────────────────
    result = {}
    for i, spot in enumerate(WIND_SPOTS):
        uid = spot_to_unique[i]
        result[spot["name"]] = unique_records[uid]

    return result
