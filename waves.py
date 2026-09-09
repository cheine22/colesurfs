"""
colesurfs — Open-Meteo Marine API Fetcher (GFS-Wave forecasts)

Requests primary + secondary + tertiary swell partitions and falls back to
the base variable set on a 400. GFS uses a fallback chain of model
identifiers (ncep_gfswave025 is current — the Marine API requires the
resolution suffix). All spots go out in one multi-location call.

EURO is NOT served from here since v1.5 — it lives in waves_cmems.py.
"""
import requests
from cache import ttl_cache, record_api_calls
from config import FORECAST_DAYS, TIMEZONE, SPOTS, m_to_ft
from wave_common import safe_float as _safe, build_swell_components, make_wave_record

MARINE_API = "https://marine-api.open-meteo.com/v1/marine"

# GFS wave model identifier chain — tried in order until one succeeds.
# ncep_gfswave025 is the correct current identifier (0.25° grid, confirmed via API docs).
# Legacy names retained as fallbacks in case of future API renames.
_GFS_MODEL_IDS  = ["ncep_gfswave025", "ncep_gfswave", "gfswave", "gfs_wave"]
_EURO_MODEL_IDS = ["ecmwf_wam025", "ecmwf_wam"]   # 025 suffix is the precise identifier

# Correct Open-Meteo variable names for swell partitions (confirmed via API docs).
# Secondary/tertiary use the prefix form, NOT the _2/_3 suffix form.
_WAVE_VARS_FULL = [
    "wave_height",                    "wave_period",                    "wave_peak_period",                "wave_direction",
    "swell_wave_height",              "swell_wave_period",              "swell_wave_direction",
    "secondary_swell_wave_height",    "secondary_swell_wave_period",    "secondary_swell_wave_direction",
    "tertiary_swell_wave_height",     "tertiary_swell_wave_period",     "tertiary_swell_wave_direction",
]

# Fallback: primary swell only + combined (for models that don't expose secondary/tertiary)
_WAVE_VARS_BASE = [
    "wave_height",       "wave_period",       "wave_peak_period",       "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
]


def _build_components(sw_h,  sw_p,  sw_d,
                       sw_h2, sw_p2, sw_d2,
                       sw_h3, sw_p3, sw_d3):
    """Build swell component list (up to 2) from swell partition fields only;
    combined-sea values ride along on the record for CSC2 only."""
    return build_swell_components([
        {"h_m": sw_h,  "p": sw_p,  "d": sw_d,  "type": "swell"},
        {"h_m": sw_h2, "p": sw_p2, "d": sw_d2, "type": "swell2"},
        {"h_m": sw_h3, "p": sw_p3, "d": sw_d3, "type": "swell3"},
    ])


def _parse_response(data) -> list:
    """Parse the Open-Meteo hourly response dict into a list of per-timestep records."""
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    n      = len(times)

    def col(key):
        return [_safe(v) for v in hourly.get(key, [None] * n)]

    # Combined (total) sea — carried on the record for CSC2, never shown as
    # swell. Open-Meteo serves no wave_peak_period for GFS-Wave (null on every
    # row), so the combined period is the mean period.
    wh  = col("wave_height");  wp  = col("wave_period");  wp_peak = col("wave_peak_period");  wd  = col("wave_direction")
    # Swell partitions — preferred source (correct Open-Meteo naming)
    sh  = col("swell_wave_height");           sp  = col("swell_wave_period");           sd  = col("swell_wave_direction")
    sh2 = col("secondary_swell_wave_height"); sp2 = col("secondary_swell_wave_period"); sd2 = col("secondary_swell_wave_direction")
    sh3 = col("tertiary_swell_wave_height");  sp3 = col("tertiary_swell_wave_period");  sd3 = col("tertiary_swell_wave_direction")

    records = []
    for i in range(n):
        comps = _build_components(
            sh[i],  sp[i],  sd[i],
            sh2[i], sp2[i], sd2[i],
            sh3[i], sp3[i], sd3[i],
        )
        # Top-level fields come from the highest-energy swell component,
        # not the combined wave_height — consistent with what is displayed.
        primary = comps[0] if comps else None

        # No combined-sea fallback (removed v1.13.1). Open-Meteo serves GFS
        # partitions for the full 10 days, so an empty `comps` means every
        # partition was 0 m — pure wind sea — and the old fallback rendered
        # that as a FUN "primary swell" at the mean period. Honest-empty, as
        # EURO has always been; csc2.train tags these rows "missing".

        # Raw direction: always include the best available swell direction even
        # when components are filtered out (period < 6s etc.), so the map can
        # still draw an arrow showing where the swell is coming from.
        raw_dir = None
        if primary:
            raw_dir = primary["direction_deg"]
        else:
            # Pick the best raw swell partition direction (prefer primary swell)
            for _rd in [sd[i], sd2[i], sd3[i], wd[i]]:
                if _rd is not None:
                    raw_dir = _rd
                    break

        records.append(make_wave_record(
            times[i], comps, primary, raw_dir,
            wh[i], wp_peak[i] or wp[i], wd[i],
        ))
    return records


@ttl_cache(ttl_seconds=3600)
def fetch_wave_forecast(lat: float, lon: float, model_key: str) -> list | None:
    """
    Fetch hourly wave forecast from Open-Meteo Marine API (single location).
    Kept as fallback for individual spot retries; primary path is fetch_all_wave_forecasts().
    """
    model_ids = _GFS_MODEL_IDS if model_key == "GFS" else _EURO_MODEL_IDS

    for model_id in model_ids:
        for var_list in [_WAVE_VARS_FULL, _WAVE_VARS_BASE]:
            params = {
                "latitude":      lat,
                "longitude":     lon,
                "hourly":        ",".join(var_list),
                "models":        model_id,
                "forecast_days": FORECAST_DAYS,
                "timezone":      TIMEZONE,
            }
            try:
                record_api_calls("wave_forecast", 1)
                r = requests.get(MARINE_API, params=params, timeout=25,
                                 headers={"User-Agent": "ColeSurfs/1.0"})
                if r.status_code == 400:
                    tag = 'full' if len(var_list) > 6 else 'base'
                    print(f"[models] {model_key}/{model_id} 400 with {tag} vars — next attempt…")
                    continue
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.HTTPError as e:
                print(f"[models] {model_key}/{model_id} HTTP {e.response.status_code}: "
                      f"{e.response.text[:200]}")
                break  # HTTP error for this model_id — try next model_id
            except Exception as e:
                print(f"[models] {model_key}/{model_id} @ ({lat},{lon}): {e}")
                return None

            if data.get("error"):
                reason = data.get("reason", "?")
                print(f"[models] Open-Meteo error {model_key}/{model_id}: {reason}")
                # Swell partition error → retry with base vars
                if any(k in reason for k in ("secondary", "tertiary", "not available", "not exist", "swell")):
                    continue
                # Model not found → try next model_id
                print(f"[models] {model_key}/{model_id} not found — trying next ID…")
                break

            times = data.get("hourly", {}).get("time", [])
            if not times:
                return None

            print(f"[models] {model_key} OK via model_id={model_id}, vars={len(var_list)}")
            return _parse_response(data)

    print(f"[models] {model_key} all attempts exhausted for ({lat},{lon})")
    return None


# ─── Batch multi-location wave forecast ──────────────────────────────────────

@ttl_cache(ttl_seconds=3600, skip_none=True)
def fetch_all_wave_forecasts(model_key: str) -> dict | None:
    """
    Fetch wave forecasts for ALL buoy spots in a single multi-location API call.
    Returns {spot_name: [per-timestep records]} or None on total failure.

    Uses comma-separated lat/lon — same pattern as wind.py's batched calls.
    Falls back to per-spot sequential calls if the batch request fails,
    so a single-spot error (e.g. one location over land) doesn't break everything.
    """
    model_ids = _GFS_MODEL_IDS if model_key == "GFS" else _EURO_MODEL_IDS

    lats = ",".join(str(s["lat"]) for s in SPOTS)
    lons = ",".join(str(s["lon"]) for s in SPOTS)
    n_spots = len(SPOTS)

    for model_id in model_ids:
        for var_list in [_WAVE_VARS_FULL, _WAVE_VARS_BASE]:
            params = {
                "latitude":      lats,
                "longitude":     lons,
                "hourly":        ",".join(var_list),
                "models":        model_id,
                "forecast_days": FORECAST_DAYS,
                "timezone":      TIMEZONE,
            }
            try:
                record_api_calls("wave_forecast_batch", n_spots)
                r = requests.get(MARINE_API, params=params, timeout=60,
                                 headers={"User-Agent": "ColeSurfs/1.0"})
                if r.status_code == 400:
                    tag = 'full' if len(var_list) > 7 else 'base'
                    print(f"[wave_batch] {model_key}/{model_id} 400 with {tag} vars — next attempt…")
                    continue
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.HTTPError as e:
                print(f"[wave_batch] {model_key}/{model_id} HTTP {e.response.status_code}: "
                      f"{e.response.text[:200]}")
                break  # try next model_id
            except Exception as e:
                print(f"[wave_batch] {model_key}/{model_id}: {e}")
                # On any network error, fall back to per-spot
                return _fallback_per_spot(model_key)

            # Open-Meteo returns a dict for 1 location, list for N>1
            if isinstance(data, dict):
                data = [data]

            if not data or not isinstance(data, list):
                continue

            # Check for API-level error
            if data[0].get("error"):
                reason = data[0].get("reason", "?")
                print(f"[wave_batch] API error {model_key}/{model_id}: {reason}")
                if any(k in reason for k in ("secondary", "tertiary", "not available", "not exist", "swell")):
                    continue   # retry with base vars
                break  # try next model_id

            # Verify we got data with time steps
            if not data[0].get("hourly", {}).get("time"):
                continue

            # Parse each location's response
            result = {}
            for i, spot in enumerate(SPOTS):
                if i < len(data):
                    try:
                        result[spot["name"]] = _parse_response(data[i])
                    except Exception as e:
                        print(f"[wave_batch] parse error for {spot['name']}: {e}")
                        result[spot["name"]] = None
                else:
                    result[spot["name"]] = None

            print(f"[wave_batch] {model_key} OK via {model_id}, "
                  f"{len(var_list)} vars, {n_spots} spots in 1 call")
            return result

    # All model IDs exhausted — fall back to per-spot
    print(f"[wave_batch] {model_key} batch exhausted, trying per-spot fallback…")
    return _fallback_per_spot(model_key)


def _fallback_per_spot(model_key: str) -> dict | None:
    """Fall back to individual per-spot fetches (sequential, each independently cached)."""
    result = {}
    any_success = False
    for s in SPOTS:
        data = fetch_wave_forecast(s["lat"], s["lon"], model_key)
        result[s["name"]] = data
        if data is not None:
            any_success = True
    return result if any_success else None
