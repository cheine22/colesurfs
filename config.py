"""
colesurfs — Configuration

Loads all region/buoy/spot data from regions.yaml at import time and exposes:
  SPOTS         — one entry per buoy region (name, buoy_id, lat, lon)
  WIND_SPOTS    — one entry per surf spot (name, lat, lon, tide info, shore_normal, etc.)
  REGION_VIEWS  — per-region map center/zoom for regional mode

Also defines the color palette, model identifiers and unit conversion
helpers. The map's wind grid geometry lives in wind_field.py.
"""
import math, os, yaml

# ─── Load regions from YAML ──────────────────────────────────────────────────
# Single source of truth for all regions, buoys, and surf spots.
# To add a new region: edit regions.yaml — no code changes needed.
_REGIONS_PATH = os.path.join(os.path.dirname(__file__), "regions.yaml")
with open(_REGIONS_PATH, "r") as _f:
    _REGIONS_RAW = yaml.safe_load(_f)

# Build SPOTS (buoys) and WIND_SPOTS (surf spots) from the YAML data.
SPOTS = []
WIND_SPOTS = []
REGION_VIEWS = {}   # {region_name: {center: [lat, lon], zoom: int}}

for _region_name, _region in _REGIONS_RAW.items():
    SPOTS.append({
        "name": _region_name,
        "buoy_id": str(_region["buoy_id"]),
        "lat": _region["buoy_lat"],
        "lon": _region["buoy_lon"],
    })
    if "map_center" in _region and "map_zoom" in _region:
        rv = {
            "center": _region["map_center"],
            "zoom": _region["map_zoom"],
        }
        if "mobile_map_center" in _region:
            rv["mobile_center"] = _region["mobile_map_center"]
        if "mobile_map_zoom" in _region:
            rv["mobile_zoom"] = _region["mobile_map_zoom"]
        REGION_VIEWS[_region_name] = rv
    for _spot in _region.get("spots", []):
        WIND_SPOTS.append({
            "name": _spot["name"],
            "lat": _spot["lat"],
            "lon": _spot["lon"],
            "buoy_region": _region_name,
            "tide_station": str(_spot["tide_station"]),
            "tide_hi_offset": _spot["tide_hi_offset"],
            "tide_lo_offset": _spot["tide_lo_offset"],
            "shore_normal": _spot["shore_normal"],
            "surfline_url": _spot["surfline_url"],
        })

# ─── Hue Mac Palette (mirrors the :root tokens in index.html) ───────────────────────────────────
HUE = {
    "bg0":        "#0d0d0f",
    "bg1":        "#131316",
    "bg2":        "#1a1a1f",
    "bg3":        "#222228",
    "bg4":        "#2a2a32",
    "border0":    "#1e1e24",
    "border1":    "#2e2e38",
    "border2":    "#3e3e4e",
    "text0":      "#e8e8f0",
    "text1":      "#a0a0b8",
    "text2":      "#606075",
    "text3":      "#404055",
    "accent":     "#7c6af7",
    "accent_dim": "#4a3faa",
    "accent_glow":"#7c6af733",
    "green":      "#3fb950",
    "green_dim":  "#1a4a22",
    "amber":      "#d29922",
    "red":        "#f85149",
}

# ─── Swell Categorization ─────────────────────────────────────────────────────
# Thresholds: edit swell-categorization-scheme.toml (or /tuner) → Refresh in app.
# Colors:     edit COLORS dict in swell_rules.py → restart.

MODEL_COLORS = {
    "EURO": HUE["accent"],   # purple
    "GFS":  HUE["green"],    # green
}

# Wind models for the per-spot forecasts (table cells, ratings, widget).
# ecmwf_ifs:    ECMWF IFS atmospheric model (Open-Meteo API default resolution)
#               1-hourly for first 90 h, 3-hourly after, 6-hourly after 144 h
# gfs_seamless: NOAA GFS seamless (hourly for d0-5, then 3-h) — matches Windy's GFS layer
WIND_MODELS = {
    "EURO": "ecmwf_ifs",
    "GFS":  "gfs_seamless",
}

# ─── Direction Helpers ────────────────────────────────────────────────────────
_CARDINALS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

def degrees_to_cardinal(deg):
    if deg is None:
        return "?"
    return _CARDINALS_16[round(float(deg) / 22.5) % 16]

def wind_to_uv(speed_ms, direction_deg):
    """Met-convention direction → U (east) / V (north) components."""
    if speed_ms is None or direction_deg is None:
        return 0.0, 0.0
    rad = math.radians(float(direction_deg))
    return round(-speed_ms * math.sin(rad), 4), round(-speed_ms * math.cos(rad), 4)

# ─── Unit Conversions ─────────────────────────────────────────────────────────
def m_to_ft(m):
    return round(m * 3.28084, 1) if m is not None else None

def ms_to_mph(ms):
    return round(ms * 2.23694, 1) if ms is not None else None

def ms_to_kts(ms):
    return round(ms * 1.94384, 1) if ms is not None else None

# ─── Forecast Settings ────────────────────────────────────────────────────────
FORECAST_DAYS = 10
TIMEZONE      = "America/New_York"

# Wave-model publication hours (UTC) — drives the dashboard run indicator,
# smart refresh, and CMEMS cache invalidation. EURO waves are capped at
# 2 cycles/day upstream: CMEMS ANFC distributes only the 00Z/12Z runs.
# EURO = CMEMS ANFC bulletins: the 00Z run lands ~08:20–09:10 UTC, the 12Z run
# ~20:40–20:55 UTC (file mtimes, 2026-09). 10Z / 21Z leave a margin; the
# warmer refetches within 30 min of these.
MODEL_UPDATE_HOURS_UTC = {
    "GFS":  [4, 10, 16, 22],
    "EURO": [10, 21],
}

# Wind-model publication hours (UTC), 4 runs/day for both atmospheric models
# (ECMWF IFS 00/06/12/18Z, Open-Meteo has each ~7 h after init). Kept separate
# from MODEL_UPDATE_HOURS_UTC so the run indicator stays truthful about wave
# runs. (The map's GRIB wind fields probe the sources directly — wind_field.py.)
WIND_UPDATE_HOURS_UTC = {
    "GFS":  [4, 10, 16, 22],
    "EURO": [1, 7, 13, 19],
}
