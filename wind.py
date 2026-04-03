"""
colesurfs — Wind Data Fetcher
  • fetch_wind_grid(model_key)              → current snapshot for map init
  • fetch_wind_forecast_grid(model_key)     → full hourly time series for hover-sync
  • fetch_spot_wind()                       → per-spot current wind for table
  • fetch_spot_wind_forecasts()             → per-spot hourly wind for WIND table row
  • fetch_region_wind_forecasts(model_key)  → hourly wind for all WIND_SPOTS (regional mode)
  • estimate_model_run(model_key)           → best guess of which model run is current

Wind model is matched to the active wave model:
  EURO → ecmwf_ifs atmospheric model
  GFS  → gfs atmospheric model

Smart caching: each fetch checks whether a new model run is likely available
since the last fetch.  If not, the cached value is returned even if the TTL
has expired (up to a hard max of 6 hours).  This avoids burning API calls
when the underlying model data hasn't changed.
"""
import time
import requests
from datetime import datetime, timezone, timedelta
from cache import ttl_cache, model_aware_cache, record_api_calls, get_cache_age
from config import (
    GRID_LATS, GRID_LONS, GRID_NY, GRID_NX,
    GRID_LA1, GRID_LO1, GRID_DX, GRID_DY,
    TIMEZONE, FORECAST_DAYS, WIND_MODELS, MODEL_UPDATE_HOURS_UTC,
    wind_to_uv, ms_to_kts, ms_to_mph, degrees_to_cardinal, SPOTS, WIND_SPOTS,
)

FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Negative cache: when a request fails, don't retry for this many seconds.
_NEGATIVE_CACHE_SEC = 600   # 10 min cooldown after API failure (e.g. 429 rate limit)
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
                # The model init time is ~6-8h before the availability time.
                # GFS: available ~4h after init. EURO: available ~7-8h after init.
                if model_key == "GFS":
                    init_offset = 4
                else:
                    init_offset = 7
                init_time = available_at - timedelta(hours=init_offset)
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


def _new_run_available_since(model_key: str, cache_age_sec: float) -> bool:
    """
    Check if a new model run has likely become available since the cache was populated.
    Returns True if we should re-fetch, False if cached data is still the latest.
    """
    if cache_age_sec is None:
        return True  # no cache → must fetch

    now = datetime.now(timezone.utc)
    cached_at = now - timedelta(seconds=cache_age_sec)
    update_hours = MODEL_UPDATE_HOURS_UTC.get(model_key, [7, 19])

    # Check if any update hour falls between cached_at and now
    for days_back in range(2):
        check_day = now - timedelta(days=days_back)
        for h in update_hours:
            update_time = check_day.replace(hour=h, minute=0, second=0, microsecond=0)
            if cached_at < update_time <= now:
                return True

    return False


# ─── Single-query helper (1 retry only) ─────────────────────────────────────

def _single_query(params: dict, model_id: str | None,
                  label: str = "wind", n_points: int = 1) -> list | None:
    """
    One Open-Meteo request for all grid points (≤100).
    Tries with model_id first, then without.  1 attempt per param set (no retry).
    Records API call count for usage tracking.
    Returns list of per-point dicts, or None.
    """
    param_sets = []
    if model_id:
        param_sets.append({**params, "models": model_id})
    param_sets.append(params)

    for p in param_sets:
        try:
            record_api_calls(label, n_points)
            r = requests.get(FORECAST_API, params=p, timeout=60,
                             headers={"User-Agent": "ColeSurfs/1.0"})
            r.raise_for_status()
            raw = r.json()
        except Exception as e:
            print(f"[{label}] ({p.get('models','default')}): {e}")
            # On rate limit (429), don't retry with fallback model — it will
            # also be rejected and just wastes quota / extends the cooldown.
            if "429" in str(e):
                return None
            continue

        if isinstance(raw, dict):
            raw = [raw]
        if not raw or not isinstance(raw, list):
            continue
        if raw[0].get("error"):
            print(f"[{label}] API error ({p.get('models','default')}): "
                  f"{raw[0].get('reason','?')}")
            break  # try next param set (without model)
        return raw
    return None


# ─── Current wind snapshot (for map init) ────────────────────────────────────
@model_aware_cache(hard_ttl=21600, model_arg_index=0)
def fetch_wind_grid(model_key: str = "EURO") -> dict | None:
    """
    Returns dict with CURRENT wind at all grid points.
    Single Open-Meteo call (grid ≤100 points).
    {u, v (flat N→S row-major), la1, lo1, nx, ny, dx, dy}
    """
    neg_key = f"wind_grid:{model_key}"
    if _is_negative_cached(neg_key):
        print(f"[wind_grid] skipping — negative cached (rate limit cooldown)")
        return None

    model_id    = WIND_MODELS.get(model_key)
    grid_points = [(lat, lon)
                   for lat in reversed(GRID_LATS)
                   for lon in GRID_LONS]
    n_pts = len(grid_points)

    lats_str = ",".join(str(p[0]) for p in grid_points)
    lons_str = ",".join(str(p[1]) for p in grid_points)

    # Try 'current' mode first (fastest), fall back to hourly
    data = _single_query({
        "latitude": lats_str, "longitude": lons_str,
        "current": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms", "timezone": TIMEZONE,
    }, model_id, label="wind_grid", n_points=n_pts)

    # If 'current' mode failed or model doesn't support it, try hourly fallback
    if not data or not data[0].get("current"):
        print("[wind_grid] 'current' mode unavailable, trying hourly fallback…")
        hourly_data = _single_query({
            "latitude": lats_str, "longitude": lons_str,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms", "timezone": TIMEZONE,
            "forecast_days": 1,
        }, model_id, label="wind_grid_hourly", n_points=n_pts)
        if hourly_data:
            data = []
            for pt in hourly_data:
                h = pt.get("hourly", {})
                spds = h.get("wind_speed_10m", [None])
                dirs = h.get("wind_direction_10m", [None])
                data.append({"current": {
                    "wind_speed_10m": spds[0] if spds else None,
                    "wind_direction_10m": dirs[0] if dirs else None,
                }})

    if not data:
        print("[wind_grid] failed")
        _set_negative_cache(neg_key)
        return None

    u_arr, v_arr = [], []
    for pt in data:
        cur = pt.get("current", {})
        u, v = wind_to_uv(cur.get("wind_speed_10m"), cur.get("wind_direction_10m"))
        u_arr.append(u)
        v_arr.append(v)

    print(f"[wind_grid] OK model={model_key}, {n_pts} pts")
    return {"u": u_arr, "v": v_arr,
            "la1": GRID_LA1, "lo1": GRID_LO1,
            "nx": GRID_NX,   "ny": GRID_NY,
            "dx": GRID_DX,   "dy": GRID_DY}


# ─── Full hourly wind forecast for the grid (for hover time-sync) ─────────────

@model_aware_cache(hard_ttl=21600, model_arg_index=0)
def fetch_wind_forecast_grid(model_key: str = "EURO") -> dict | None:
    """
    Single Open-Meteo call for all grid points (≤100 points).
    model_key selects the atmospheric model.
    Uses smart caching: skips fetch if no new model run is available.
    Negative caching: on failure, waits 10 min before retrying.

    Returns:
      {
        "times":       ["2026-03-23T12:00", ...],
        "u_by_time":   [[u_p0, u_p1, ..., u_pN], ...],
        "v_by_time":   [[v_p0, ...], ...],
        "grid":        {la1, lo1, nx, ny, dx, dy}
      }
    """
    neg_key = f"wind_forecast:{model_key}"
    if _is_negative_cached(neg_key):
        print(f"[wind_forecast] skipping — negative cached (rate limit cooldown)")
        return None

    model_id    = WIND_MODELS.get(model_key)
    grid_points = [(lat, lon)
                   for lat in reversed(GRID_LATS)
                   for lon in GRID_LONS]
    n_pts = len(grid_points)

    lats_str = ",".join(str(p[0]) for p in grid_points)
    lons_str = ",".join(str(p[1]) for p in grid_points)

    print(f"[wind_forecast] fetching {n_pts} pts in 1 query…")
    data = _single_query({
        "latitude": lats_str, "longitude": lons_str,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "forecast_days": FORECAST_DAYS,
        "timezone": TIMEZONE,
    }, model_id, label="wind_forecast", n_points=n_pts)

    if not data:
        print("[wind_forecast] failed — no wind data")
        _set_negative_cache(neg_key)
        return None

    times = data[0].get("hourly", {}).get("time", [])
    if not times:
        print("[wind_forecast] no time steps in response")
        return None

    n_times = len(times)
    print(f"[wind_forecast] OK model={model_key}, "
          f"{n_times} timesteps × {len(data)} grid points")

    u_by_time, v_by_time = [], []
    for t_idx in range(n_times):
        u_row, v_row = [], []
        for p_data in data:
            h      = p_data.get("hourly", {})
            speeds = h.get("wind_speed_10m",     [None] * n_times)
            dirs   = h.get("wind_direction_10m", [None] * n_times)
            spd  = speeds[t_idx] if t_idx < len(speeds) else None
            dirn = dirs[t_idx]   if t_idx < len(dirs)   else None
            u, v = wind_to_uv(spd, dirn)
            u_row.append(u)
            v_row.append(v)
        u_by_time.append(u_row)
        v_by_time.append(v_row)

    return {
        "times":     times,
        "u_by_time": u_by_time,
        "v_by_time": v_by_time,
        "grid": {
            "la1": GRID_LA1, "lo1": GRID_LO1,
            "nx":  GRID_NX,  "ny":  GRID_NY,
            "dx":  GRID_DX,  "dy":  GRID_DY,
        },
    }


# ─── Per-spot current wind ────────────────────────────────────────────────────
@ttl_cache(ttl_seconds=3600)
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
    except Exception:
        return None
    cur  = d.get("current", {})
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


# ─── Per-spot hourly wind forecast (for WIND table row) ───────────────────────
@ttl_cache(ttl_seconds=3600)
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
def fetch_region_wind_forecasts(model_key: str = "EURO") -> dict | None:
    """
    Hourly wind + gust forecast for all WIND_SPOTS, respecting model_key.
    Returns {spot_name: [{time, speed_mph, direction_deg, direction_cardinal,
                          gust_mph, gust_cardinal}, ...]}
    Uses WIND_MODELS[model_key] atmospheric model (same as wind grid).
    Falls back to API default if the requested model fails.

    Deduplicates spots that share the same lat/lon (e.g. spots appearing in
    multiple regions) so the API call uses only unique coordinates, saving
    quota and avoiding 429 rate-limit errors.
    """
    if not WIND_SPOTS:
        return {}

    neg_key = f"region_wind:{model_key}"
    if _is_negative_cached(neg_key):
        print(f"[region_wind] skipping — negative cached (rate limited recently)")
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

    base_params = {
        "latitude":        lats,
        "longitude":       lons,
        "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms",
        "forecast_days":   FORECAST_DAYS,
        "timezone":        TIMEZONE,
    }

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


# ─── Wire model-run-aware caching ────────────────────────────────────────────
# Set the checker on model_aware_cache-decorated functions so they can
# decide whether to re-fetch based on model run availability.
fetch_wind_grid._new_run_checker = _new_run_available_since
fetch_wind_forecast_grid._new_run_checker = _new_run_available_since
