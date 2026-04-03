"""
colesurfs — Configuration

Loads all region/buoy/spot data from regions.yaml at import time and exposes:
  SPOTS         — one entry per buoy region (name, buoy_id, lat, lon)
  WIND_SPOTS    — one entry per surf spot (name, lat, lon, tide info, shore_normal, etc.)
  REGION_VIEWS  — per-region map center/zoom for regional mode

Also defines the color palette, wind speed bands, model identifiers, unit
conversion helpers, and the wind-map grid geometry.
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

# ─── Hue Mac Palette (exact from style.css) ───────────────────────────────────
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

# ─── Wind Speed Scale ─────────────────────────────────────────────────────────
# Labels shown below the wind map. Units: mph.
# Colors should match the leaflet-velocity colorScale (7 bands).
# The velocity layer maps 0 → 20 m/s = 0 → 45 mph linearly across
# the 7-color scale: calm → light → moderate → fresh → strong → near-gale → gale+
#
# Format: (min_mph, max_mph_or_None, bg_color, text_color)
# Use None for the last band's max to display "45+".
WIND_BANDS = [
    (0,   5,    "#1e1e2e", "#404055"),
    (5,  10,   "#3a2e88", "#c0b8ff"),
    (10,  25,   "#3fb950", "#ffffff"),
    (25,  45,   "#d29922", "#ffffff"),
    (45,  None, "#f85149", "#ffffff"),
]

# ─── Swell Categorization ─────────────────────────────────────────────────────
# Thresholds: edit swell-categorization-scheme.toml → Refresh in app (or restart).
# Colors:     edit COLORS dict in swell_rules.py → restart.
# See also: Accessory Pages/swell-categorization-scheme.html for a visual reference.

# ─── Open-Meteo Model Identifiers ─────────────────────────────────────────────
MODELS = {
    "EURO": "ecmwf_wam",
    "GFS":  "ncep_gfswave",
}

MODEL_COLORS = {
    "EURO": HUE["accent"],   # purple
    "GFS":  HUE["green"],    # green
}

# Wind models (for wind forecast grid + regional spot forecasts).
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

def degrees_to_travel_arrow(deg):
    """Arrow showing where waves are TRAVELING TO (opposite of 'from' direction)."""
    if deg is None:
        return "·"
    travel = (float(deg) + 180) % 360
    arrows = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
    return arrows[round(travel / 45) % 8]

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

# ─── Wind Map Grid: Western Atlantic basin ───────────────────────────────────
# 144 grid points (12 rows × 12 cols) — fits in ONE Open-Meteo API call.
# Covers 5°N–49°N × 83°W–39°W (East Coast + NW Atlantic + Caribbean edge).
# 4° spacing provides smooth wind particle animation while staying within
# free API limits.
WIND_MAP_CENTER = [41.2, -71.5]
WIND_MAP_ZOOM   = 8

GRID_LATS = [5.0 + i * 4.0 for i in range(12)]    # 5°N → 49°N, 4° step  (12 rows)
GRID_LONS = [-83.0 + i * 4.0 for i in range(12)]   # 83°W → 39°W, 4° step (12 cols)
GRID_NY   = len(GRID_LATS)   # 12
GRID_NX   = len(GRID_LONS)   # 12
GRID_DX   = 4.0
GRID_DY   = 4.0
GRID_LA1  = float(GRID_LATS[-1])   # 49.0 — northernmost (grid starts NW)
GRID_LO1  = float(GRID_LONS[0])    # -83.0 — westernmost

# ─── Forecast Settings ────────────────────────────────────────────────────────
FORECAST_DAYS = 10
TIMEZONE      = "America/New_York"

MODEL_UPDATE_HOURS_UTC = {
    "GFS":  [4, 10, 16, 22],
    "EURO": [7, 19],
}
