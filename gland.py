"""
colesurfs — G-Land (Grajagan, East Java) data module

A standalone data layer for the /gland page. G-Land sits outside every
assumption the main dashboard makes: there is no NDBC buoy within thousands
of kilometres, and NOAA CO-OPS has no Indonesian tide station. So nothing
here routes through regions.yaml / buoy.py / tide.py — it is its own set of
sources, chosen because they actually resolve at 8.7°S 114.3°E:

  waves   GFS-Wave via Open-Meteo Marine (real swell partitions) and
          ECMWF-WAM via Copernicus Marine (waves_cmems.fetch_cmems_point,
          which already takes an arbitrary lat/lon)
  tide    Open-Meteo Marine `sea_level_height_msl` — a global tide model.
          CO-OPS stops at the US border; the nearest working IOC gauge on
          this coast is Prigi, 292 km up the coast, and Indonesian tides are
          amphidromic enough that borrowing it would be worse than a model.
  wind    Open-Meteo forecast API at the point
  upstream  AODN/IMOS near-real-time wave buoys down the Western Australian
          coast — the only in-situ swell measurements anywhere near G-Land's
          storm track (see UPSTREAM_BUOYS for the honest caveat on geometry)

The G-Land-specific logic — swell-window scoring and per-section section
ranking — lives in this module rather than the template, so the page and any
future consumer see the same numbers.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import swell_rules
from cache import ttl_cache, record_api_calls
from config import m_to_ft

try:
    import tomllib
except ImportError:                       # Python < 3.11
    import tomli as tomllib


# ── G-Land-only swell categories ────────────────────────────────────────────
# G-Land gets its own FLAT..MONSTRO thresholds, in its own TOML, with its own
# cache. It shares NOTHING mutable with swell_rules beyond the category names
# and colours — the site-wide scheme is tuned for NY/NJ beach breaks and a
# 6 ft 17 s Indian Ocean point break is a different animal. Editing one must
# never move the other, which is why this does not call swell_rules.load_bands()
# or touch its module-level cache.
GLAND_TOML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "gland-swell-categorization.toml")
_GLAND_RULES = None

# G-Land's ladder, low -> high, for legends and ordering. DREAMY sits at the
# top because it is the one everybody is actually chasing, not because it is
# the biggest. BIG and DREAMY do not exist site-wide.
GLAND_CATEGORIES = ["FLAT", "WEAK", "FUN", "SOLID", "FIRING", "BIG",
                    "HECTIC", "DREAMY"]

# Colours: reuse the site palette where the name matches, then add the two
# G-Land-only categories. swell_rules.COLORS is NOT mutated — the rest of the
# site must not learn about DREAMY or BIG.
GLAND_COLORS = {k: dict(v) for k, v in swell_rules.COLORS.items()}
GLAND_COLORS["DREAMY"] = dict(swell_rules.COLORS["MONSTRO"])   # as requested
# The site's HECTIC violet (#7c6af7, hue 248) sat only 22 deg from MONSTRO's
# purple, so on the G-Land ladder HECTIC and DREAMY read as the same colour.
# DREAMY keeps the purple; HECTIC moves to magenta (hue ~330), which is >=34 deg
# from every other category here. Site-wide HECTIC is untouched — this dict is
# a copy.
GLAND_COLORS["HECTIC"] = dict(dark_bg="#2b0a1c", dark_text="#ff4fa3",
                              light_bg="#fbd9ec", light_text="#a3105e")
GLAND_COLORS["BIG"] = dict(dark_bg="#2a2a32", dark_text="#d8d8e4",
                           light_bg="#c6c6d2", light_text="#26262e")


def _bound(v):
    """TOML bound -> float, with "inf" meaning unbounded."""
    if isinstance(v, str):
        return float("inf") if v.strip().lower() == "inf" else float(v)
    return float(v)


def load_gland_rules(force: bool = False):
    """Ordered match rules. First hit wins, so order is priority."""
    global _GLAND_RULES
    if _GLAND_RULES is not None and not force:
        return _GLAND_RULES
    with open(GLAND_TOML, "rb") as f:
        data = tomllib.load(f)
    rules = []
    for r in data.get("rule", []):
        entry = {
            "category": r["category"],
            "period": [_bound(x) for x in r.get("period", [0, "inf"])],
            "height": [_bound(x) for x in r.get("height", [0, "inf"])],
        }
        if "direction" in r:
            entry["direction"] = [_bound(x) for x in r["direction"]]
        rules.append(entry)
    _GLAND_RULES = rules
    return rules


def reload_gland_rules():
    return load_gland_rules(force=True)


def _in_range(v, lo, hi):
    # Inclusive low, exclusive high — except an unbounded high.
    if v is None:
        return False
    return v >= lo and (hi == float("inf") or v < hi)


def categorize_gland(height_ft, period_s, direction_deg=None):
    """First matching rule wins; no match falls through to FLAT.

    `direction_deg` is optional so callers that genuinely have no direction
    still work — but a rule carrying a direction range can never match when it
    is absent, which is the safe failure (DREAMY is a claim, not a default).
    """
    if height_ft is None or period_s is None:
        return "FLAT"
    for r in load_gland_rules():
        if not _in_range(period_s, *r["period"]):
            continue
        if not _in_range(height_ft, *r["height"]):
            continue
        if "direction" in r:
            if direction_deg is None or not _in_range(direction_deg, *r["direction"]):
                continue
        return r["category"]
    return "FLAT"


def gland_rules_payload():
    """JSON-friendly rules for the page and the tuner ("inf" stays a string)."""
    def b(v):
        return "inf" if v == float("inf") else v
    out = []
    for r in load_gland_rules():
        e = {"category": r["category"],
             "period": [b(x) for x in r["period"]],
             "height": [b(x) for x in r["height"]]}
        if "direction" in r:
            e["direction"] = [b(x) for x in r["direction"]]
        out.append(e)
    return out

# ── The spot ────────────────────────────────────────────────────────────────
# Coordinates match Surfline's G-Land spot pin (5842041f4e65fad6a7708d30),
# verified against Esri World Imagery — the pin sits on the breaking reef edge
# at the Moneytrees section.
GLAND_LAT = -8.7269
GLAND_LON = 114.34882
GLAND_TZ = "Asia/Jakarta"          # WIB, UTC+7
# Same spot id the coordinates came from; the main dashboard keeps its
# equivalent in regions.yaml as `surfline_url`.
SURFLINE_URL = ("https://www.surfline.com/surf-report/g-land/"
                "5842041f4e65fad6a7708d30")
FORECAST_DAYS = 10

# ── Where swell is sampled ──────────────────────────────────────────────────
# NOT at the spot pin. Surfline reports *deepwater* swell offshore of a spot,
# and sampling a global wave model at the pin gives something else entirely:
# the Blambangan peninsula wraps around G-Land, so the inshore model cells are
# land-shadowed. Measured 2026-07-28, same hour, GFS-Wave 0.25°:
#
#   -8.750, 114.250  (cell the pin snaps to)   0.96 m @ 11.3 s from 180°
#   -9.000, 114.250  (one cell south, open)    0.92 m @ 15.3 s from 209°
#
# Same height, but the inshore cell has lost the long-period SSW energy —
# that is shadowing and refraction, not a different swell. CMEMS shows the
# same gradient more gently (14.2 s at the pin vs 15.7 s offshore) because its
# 1/12° grid resolves the coast better.
#
# So both models sample this open-ocean node ~32 km SSW of the point, square
# in the swell window. It also keeps GFS-vs-EURO an apples-to-apples
# comparison, since both read the identical location.
SWELL_NODE_LAT = -9.00
SWELL_NODE_LON = 114.25
SWELL_NODE_KM = 32                 # distance from the spot, for the UI note

MARINE_API = "https://marine-api.open-meteo.com/v1/marine"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
AODN_WFS = "https://geoserver-123.aodn.org.au/geoserver/ows"

_GFS_MODEL_IDS = ["ncep_gfswave025", "ncep_gfswave", "gfswave"]

_WAVE_VARS = [
    "wave_height", "wave_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "secondary_swell_wave_height", "secondary_swell_wave_period",
    "secondary_swell_wave_direction",
    "tertiary_swell_wave_height", "tertiary_swell_wave_period",
    "tertiary_swell_wave_direction",
]

# ── Swell window ────────────────────────────────────────────────────────────
# Surfline (Mechanics of G-Land): "SW to SSW (around 240-200°) is generally
# the best swell direction for the various sections of the wave." Westerly
# swell still works but focuses on Kongs; straight south is the extreme limit
# G-Land can receive (Swellnet). Outside 165-285° nothing reaches the point —
# beyond that is SE trade windsea, which is *offshore* here and must never be
# mistaken for surf.
#
# SWELL_BANDS splits the window by which section each angle actually feeds,
# and drives the direction dial in the UI.
WINDOW_CORE = (190.0, 250.0)       # full credit
WINDOW_EDGE = (165.0, 285.0)       # tapers to zero at the edges

# Bands OVERLAP by design: 205-210° feeds Speedies and Moneytrees at once, so
# `ring` splits them radially in the dial (outer vs inner) and the overlap
# lights both. Anything consuming these must handle a direction matching more
# than one band.
SWELL_BANDS = [
    {"lo": 165, "hi": 190, "key": "outer", "label": "Outer window", "ring": "full",
     "note": "Extreme southerly. Sweeps the top of the point; feels out to sea."},
    {"lo": 190, "hi": 210, "key": "speedies", "label": "Speedies", "ring": "outer",
     "note": "The bigger SSW that lights up Launching Pad into Speedies."},
    {"lo": 205, "hi": 250, "key": "moneytrees", "label": "Moneytrees", "ring": "inner",
     "note": "The classic SW-SSW band. Best all-round angle for the point."},
    {"lo": 250, "hi": 285, "key": "outer", "label": "Outer window", "ring": "full",
     "note": "Westerly. Biggest surf, but focuses into Kongs — inside drops off."},
]

# Period below this is trade windsea, not Indian Ocean groundswell.
MIN_GROUNDSWELL_PERIOD = 9.0


def _dir_in_window(deg):
    """0..1 how well a swell direction feeds the point. 1.0 across the
    SW-SSW core, tapering linearly to 0 at the edges of the window."""
    if deg is None:
        return 0.0
    d = deg % 360.0
    lo_c, hi_c = WINDOW_CORE
    lo_e, hi_e = WINDOW_EDGE
    if lo_c <= d <= hi_c:
        return 1.0
    if lo_e <= d < lo_c:
        return (d - lo_e) / (lo_c - lo_e)
    if hi_c < d <= hi_e:
        return (hi_e - d) / (hi_e - hi_c)
    return 0.0


def _safe(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _swell_score(height_ft, period_s, direction_deg):
    """Rank a swell partition by how much surf it will actually make at
    G-Land. Energy ∝ H²T, weighted by swell-window fit, with a hard floor on
    period so short-period trade slop can never win."""
    h, p, d = _safe(height_ft), _safe(period_s), _safe(direction_deg)
    if h is None or p is None or d is None or h <= 0:
        return 0.0
    if p < MIN_GROUNDSWELL_PERIOD:
        return 0.0
    w = _dir_in_window(d)
    if w <= 0:
        return 0.0
    return (h ** 2) * p * w


def pick_gland_swell(components):
    """From a list of swell partitions pick the one that will actually break
    at G-Land.

    This is the crux of forecasting this place from raw model output. The
    dashboard's usual energy-sort puts the biggest partition first, and at
    G-Land in the dry season the biggest partition is routinely a 7-8 s SE
    trade windsea that the point never sees — while the wave everyone flew
    here for is a smaller-looking 16 s line out of the SSW. Sorting by raw
    energy would show 5 ft when the surf is 2.4 ft of long-period gold, or
    vice versa. So we sort by window-weighted energy instead.

    Returns (best_component_or_None, all_components_scored).
    """
    scored = []
    for c in components or []:
        c = dict(c)
        c["gland_score"] = _swell_score(
            c.get("height_ft"), c.get("period_s"), c.get("direction_deg"))
        c["in_window"] = c["gland_score"] > 0
        scored.append(c)
    ranked = [c for c in scored if c["gland_score"] > 0]
    ranked.sort(key=lambda c: c["gland_score"], reverse=True)
    return (ranked[0] if ranked else None), scored


# ── The lineup ──────────────────────────────────────────────────────────────
# Section order is outside-in, the way a swell meets the reef: Kongs at the
# top of the point, then Moneytrees, then Launching Pads into Speed Reef,
# then Chickens, with Tiger Tracks a walk further down the beach.
#
# `faces` is the seaward-facing direction of that stretch of reef; `offshore`
# is its reciprocal, i.e. the wind direction that grooms it. Surfline's
# description drives these: the tip of the point faces due west and the
# shoreline swings to NNW as the wave wraps in, so the top of the point wants
# an easterly trade and the inside sections want a more southerly one, with
# ESE the best overall compromise.
#
# `size_ft` is the surfable range in *Hawaiian* feet as reported by Surfline
# and the camps; the page converts to face height for display.
SECTIONS = [
    {
        "key": "kongs", "prestige": 0.85,
        "name": "Kongs", "abbr": "Kongs",
        "lat": -8.72950, "lon": 114.34600,
        "blurb": "Top of the point. Long, sectiony, rippable — the saviour "
                 "when everything inside is too small.",
        "faces": 262, "offshore": 82,
        "size_ft": [2, 12], "min_face_ft": 3, "ideal_face_ft": [5, 14],
        "best_dir": [250, 285],      # west-angled swell focuses here
        "best_period": [12, 20],
        "tide": "any",
        "tide_note": "Surfable at low tide up to the limit of your nerve.",
        "detail": "Picks up more swell than anything else on the reef and is "
                  "rarely under 3 ft. Ledges and gets hollow on a west-angled "
                  "swell; south swells peel and section, bending out to sea.",
    },
    {
        "key": "moneytrees", "prestige": 0.97,
        "name": "Moneytrees", "abbr": "MTs",
        "lat": -8.72751, "lon": 114.34903,
        "blurb": "The Out Front wave. Long walled tube that holds its shape "
                 "end to end.",
        "faces": 290, "offshore": 110,
        "size_ft": [2, 10], "min_face_ft": 4, "ideal_face_ft": [6, 16],
        "best_dir": [205, 250],
        "best_period": [14, 20],
        "tide": "mid-high",
        "tide_note": "Rideable through the tides; prefers higher water, but "
                     "it has become the consensus that it's more fun shallower.",
        "detail": "Stands up on a well-defined reef ledge and drives down the "
                  "line with a lot of positive angle. High almond tube. Not a "
                  "place to get caught inside.",
    },
    {
        "key": "speedies", "prestige": 1.00,
        "name": "Speedies", "abbr": "Speedies",
        "lat": -8.72653, "lon": 114.35150,
        "blurb": "The main event. Straight, shallow, fast — 20-second tubes.",
        "faces": 315, "offshore": 135,
        # Surfline has it "rideable from about 2 to 8 feet+ (Hawaiian)" but
        # "usually needs larger swells" — so it switches on around 6 ft faces
        # and only really comes into its own from 9 ft up.
        "size_ft": [2, 8], "min_face_ft": 6, "ideal_face_ft": [9, 18],
        "best_dir": [190, 210],      # wants a bigger SSW; overlaps MTs at 205-210
        "best_period": [15, 22],
        "tide": "high",
        "tide_note": "Needs high water — and on the biggest days it wants the "
                     "highest water of the month, i.e. a spring tide.",
        "detail": "On a bigger SSW the outer Launching Pads reef switches on "
                  "and rolls you into Speed Reef: a 200-300 m barrel over dry "
                  "reef. Needs a solid trade to hold the door open. This is "
                  "where people get the wave of their life, and where people "
                  "get hurt.",
    },
    {
        "key": "chickens", "prestige": 0.62,
        "name": "Chickens", "abbr": "Chickens",
        "lat": -8.72478, "lon": 114.35643,
        "blurb": "Softer end section past the harbour channel.",
        "faces": 330, "offshore": 150,
        "size_ft": [2, 6], "min_face_ft": 3, "ideal_face_ft": [4, 9],
        "best_dir": [210, 260],
        "best_period": [10, 18],
        "tide": "mid-high",
        "tide_note": "Forgiving through most of the tide.",
        "detail": "A slightly lame little left just past the boat channel — "
                  "but the friendliest water on the reef when the point is "
                  "maxing.",
    },
    {
        "key": "tigertracks", "prestige": 0.60,
        "name": "Tiger Tracks", "abbr": "TTs",
        "lat": -8.71970, "lon": 114.36630,
        "blurb": "Peaky outbreak 20-30 min walk further down the beach.",
        "faces": 340, "offshore": 160,
        "size_ft": [2, 6], "min_face_ft": 3, "ideal_face_ft": [4, 9],
        "best_dir": [210, 260],
        "best_period": [10, 18],
        "tide": "mid",
        "tide_note": "Mid tide; shifty either side of it.",
        "detail": "Small peaky reef well down the beach. Worth the walk when "
                  "the main reef is crowded.",
    },
]

# The breaking reef edge, traced off Esri World Imagery (2026-07) from the
# south-west tip of the point north-east into Grajagan Bay. Swell arrives from
# the SW, wraps the tip, and peels left along this line. Drawn on the map so
# the sections read as points on one continuous wave rather than pins.
REEF_LINE = [
    [-8.73980, 114.33620], [-8.73639, 114.33817], [-8.73370, 114.34015],
    [-8.73190, 114.34237], [-8.73039, 114.34459], [-8.72883, 114.34681],
    [-8.72751, 114.34903], [-8.72653, 114.35150], [-8.72565, 114.35396],
    [-8.72478, 114.35643], [-8.72346, 114.35939], [-8.72165, 114.36285],
    [-8.71970, 114.36630],
]

# The opening in the reef that boats use to land on the beach. Speedies ends
# just short of it; Chickens starts beyond it.
HARBOUR_CHANNEL = [-8.72587, 114.35413]
POINT_TIP = [-8.73980, 114.33620]

# Hawaiian feet → approximate face feet. The camps and Surfline quote G-Land
# in Hawaiian scale; the models and this page work in face height.
HAWAIIAN_TO_FACE = 2.0


# ── Upstream in-situ swell ──────────────────────────────────────────────────
# There is no wave buoy in G-Land's swell window. NDBC 53401 (the one
# Indonesian station) was a DART tsunami buoy 970 km NW and was
# disestablished in 2008. The nearest in-situ *wave* measurements are the
# AODN/IMOS + WA Dept. of Transport buoys down the Western Australian coast,
# 2,300-2,900 km away.
#
# Honest caveat, which the page states plainly: these are NOT ray-trace
# intercepts for a typical G-Land swell. A SW swell passing Cape Naturaliste
# is travelling NE into the Australian coast, not north to Java; G-Land's
# 200-240° rays run up the Indian Ocean several hundred km west of the WA
# shelf. What these buoys *do* give you is ground truth that a Southern Ocean
# swell event is real, and a measured period for it, 2-3 days before it would
# reach Java. They sample the same storm track (Kerguelen-to-Heard lows
# tracking west to east) that feeds G-Land. Treat them as a storm-track
# sentinel line, not as an upstream reading of your swell.
#
# `bearing_to_gland` is the great-circle bearing from the buoy to G-Land: the
# closer to 360°, the more nearly a swell passing that buoy is heading
# straight at Java, and the more literal the reading becomes.
UPSTREAM_BUOYS = [
    {"site": "Cape Naturaliste", "lat": -33.525, "lon": 114.770},
    {"site": "Rottnest Island",  "lat": -32.100, "lon": 115.400},
    {"site": "Hillarys",         "lat": -31.700, "lon": 115.619},
    {"site": "Jurien Bay Offshore", "lat": -30.361, "lon": 114.948},
    {"site": "Dongara Offshore", "lat": -29.279, "lon": 114.863},
    {"site": "Coral Bay 0 2",    "lat": -25.311, "lon": 114.277},
    {"site": "Tantabiddi",       "lat": -23.227, "lon": 114.276},
    {"site": "Esperance",        "lat": -34.000, "lon": 121.900},
    {"site": "Bunbury",          "lat": -33.335, "lon": 115.402},
]


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    r = 6371.0
    p = math.pi / 180.0
    dlat = (b_lat - a_lat) * p
    dlon = (b_lon - a_lon) * p
    h = (math.sin(dlat / 2) ** 2
         + math.cos(a_lat * p) * math.cos(b_lat * p) * math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def bearing_deg(a_lat, a_lon, b_lat, b_lon):
    p = math.pi / 180.0
    dlon = (b_lon - a_lon) * p
    y = math.sin(dlon) * math.cos(b_lat * p)
    x = (math.cos(a_lat * p) * math.sin(b_lat * p)
         - math.sin(a_lat * p) * math.cos(b_lat * p) * math.cos(dlon))
    return (math.atan2(y, x) / p) % 360.0


def swell_travel_hours(distance_km, period_s):
    """Deep-water group speed for a given period: cg = gT/4π."""
    if not period_s or period_s <= 0:
        return None
    cg_ms = 9.81 * period_s / (4 * math.pi)
    return distance_km * 1000.0 / cg_ms / 3600.0


# ── Wind ────────────────────────────────────────────────────────────────────
def wind_for_section(section, speed_kt, dir_deg):
    """Rate the wind for one section of the reef.

    G-Land's whole character is that the trade is offshore at the point, and
    *which* trade angle is clean changes as you move down the reef — easterly
    grooms the top, more southerly grooms the inside, ESE is the best overall.
    So there is no single shore-normal for this spot; each section gets its
    own.
    """
    s, d = _safe(speed_kt), _safe(dir_deg)
    if s is None or d is None:
        return {"rating": None, "offset": None}
    # Angular difference between the wind and this section's ideal offshore.
    off = abs(((d - section["offshore"] + 180) % 360) - 180)
    if s < 3:
        rating = "GLASSY"
    elif off <= 45:
        rating = "GROOMED" if s <= 14 else "STRONG OFFSHORE"
    elif off <= 80:
        rating = "CLEAN" if s <= 12 else "TEXTURED"
    elif off <= 115:
        rating = "TEXTURED" if s <= 10 else "CHOPPY"
    else:
        rating = "ONSHORE" if s > 6 else "TEXTURED"
    return {"rating": rating, "offset": round(off)}


# ── Moon / spring tides ─────────────────────────────────────────────────────
def moon_phase(dt_epoch):
    """Fraction of the synodic month elapsed (0 = new, 0.5 = full).
    Simple mean-lunation model — good to about half a day, which is all the
    page needs to flag a spring-tide window for Speedies."""
    synodic = 29.530588853
    known_new = 947182440.0     # 2000-01-06 18:14 UTC, a new moon
    days = (dt_epoch - known_new) / 86400.0
    return (days % synodic) / synodic


def moon_label(phase):
    if phase < 0.03 or phase > 0.97:
        return "New"
    if phase < 0.22:
        return "Waxing crescent"
    if phase < 0.28:
        return "First quarter"
    if phase < 0.47:
        return "Waxing gibbous"
    if phase < 0.53:
        return "Full"
    if phase < 0.72:
        return "Waning gibbous"
    if phase < 0.78:
        return "Last quarter"
    return "Waning crescent"


def is_spring_tide(phase):
    """Springs run around new and full moon."""
    return phase < 0.10 or phase > 0.90 or 0.40 < phase < 0.60


# ── Fetchers ────────────────────────────────────────────────────────────────
@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_gfs_waves():
    """GFS-Wave partitions at G-Land via Open-Meteo Marine."""
    for model_id in _GFS_MODEL_IDS:
        params = {
            "latitude": SWELL_NODE_LAT, "longitude": SWELL_NODE_LON,
            "hourly": ",".join(_WAVE_VARS),
            "models": model_id,
            "forecast_days": FORECAST_DAYS,
            "timezone": GLAND_TZ,
        }
        try:
            r = requests.get(MARINE_API, params=params, timeout=30)
            record_api_calls(1)
            if r.status_code != 200:
                continue
            return _parse_marine(r.json(), "GFS")
        except Exception as e:
            print(f"[gland] GFS {model_id} {type(e).__name__}: {e}")
            continue
    return None


def _parse_marine(payload, model):
    h = payload.get("hourly") or {}
    times = h.get("time") or []
    out = []
    for i, t in enumerate(times):
        def g(key):
            arr = h.get(key)
            return _safe(arr[i]) if arr and i < len(arr) else None

        comps = []
        for pref, label in (("swell_wave", "swell"),
                            ("secondary_swell_wave", "swell2"),
                            ("tertiary_swell_wave", "swell3")):
            hm, ps, dd = g(f"{pref}_height"), g(f"{pref}_period"), g(f"{pref}_direction")
            if hm is None or ps is None or dd is None or hm <= 0:
                continue
            comps.append({
                "height_ft": round(m_to_ft(hm), 1),
                "period_s": round(ps, 1),
                "direction_deg": round(dd, 1),
                "type": label,
            })
        best, scored = pick_gland_swell(comps)
        out.append({
            "time": t,
            "model": model,
            "components": scored,
            "gland_swell": best,
            "combined_height_ft": round(m_to_ft(g("wave_height")), 1) if g("wave_height") else None,
            "combined_period_s": g("wave_period"),
            "combined_direction_deg": g("wave_direction"),
        })
    return out


@ttl_cache(ttl_seconds=3600, skip_none=True)
def fetch_euro_waves():
    """ECMWF-WAM partitions at G-Land via Copernicus Marine.

    Reuses the existing CMEMS point fetcher, which already accepts an
    arbitrary lat/lon and runs the shared processing pipeline (Tm01x1.20,
    5 s filter, energy-sorted top-2). We then re-rank its components through
    the G-Land window so both models are compared on the same quantity.
    """
    try:
        from waves_cmems import fetch_cmems_point
        rows = fetch_cmems_point(SWELL_NODE_LAT, SWELL_NODE_LON)
    except Exception as e:
        print(f"[gland] CMEMS {type(e).__name__}: {e}")
        return None
    if not rows:
        return None
    out = []
    for r in rows:
        best, scored = pick_gland_swell(r.get("components") or [])
        out.append({
            "time": r.get("time"),
            "model": "EURO",
            "components": scored,
            "gland_swell": best,
            "combined_height_ft": round(m_to_ft(r["combined_wave_height_m"]), 1)
                                  if r.get("combined_wave_height_m") else None,
            "combined_period_s": r.get("combined_wave_period_s"),
            "combined_direction_deg": r.get("combined_wave_direction_deg"),
        })
    return out


@ttl_cache(ttl_seconds=6 * 3600, skip_none=True)
def fetch_gland_tide():
    """Hourly tide height at G-Land from Open-Meteo's global tide model,
    plus derived highs/lows and a spring-tide flag per day."""
    params = {
        "latitude": GLAND_LAT, "longitude": GLAND_LON,
        "hourly": "sea_level_height_msl",
        "forecast_days": FORECAST_DAYS,
        "timezone": GLAND_TZ,
    }
    try:
        r = requests.get(MARINE_API, params=params, timeout=30)
        record_api_calls(1)
        if r.status_code != 200:
            return None
        h = (r.json().get("hourly") or {})
    except Exception as e:
        print(f"[gland] tide {type(e).__name__}: {e}")
        return None

    times = h.get("time") or []
    vals = h.get("sea_level_height_msl") or []
    series = []
    for t, v in zip(times, vals):
        v = _safe(v)
        series.append({"time": t, "height_m": v,
                       "height_ft": round(m_to_ft(v), 1) if v is not None else None})

    # Local extrema → high/low events.
    hilo = []
    for i in range(1, len(series) - 1):
        a, b, c = series[i - 1]["height_m"], series[i]["height_m"], series[i + 1]["height_m"]
        if a is None or b is None or c is None:
            continue
        if b >= a and b >= c and not (a == b == c):
            hilo.append({"time": series[i]["time"], "type": "H",
                         "height_ft": series[i]["height_ft"]})
        elif b <= a and b <= c and not (a == b == c):
            hilo.append({"time": series[i]["time"], "type": "L",
                         "height_ft": series[i]["height_ft"]})

    heights = [s["height_m"] for s in series if s["height_m"] is not None]
    return {
        "series": series,
        "hilo": hilo,
        "range_m": round(max(heights) - min(heights), 2) if heights else None,
        "source": "Open-Meteo global tide model (no CO-OPS/IOC gauge at G-Land)",
    }


# ── Harmonic tide model ─────────────────────────────────────────────────────
# Open-Meteo's tide field only runs ~9 days ahead, so a multi-year lookup is
# impossible from the forecast endpoint. But tides are the sum of a handful of
# astronomical constituents, and the SAME endpoint serves hourly history back
# to 1940 — so we fit constituents to a couple of years of history once, then
# predict any date, past or future.
#
# Speeds are degrees per mean solar hour. The set below resolves cleanly in a
# 2-year record (the tightest pair, K1/P1, needs ~half a year by the Rayleigh
# criterion).
TIDE_CONSTITUENTS = {
    "M2": 28.9841042, "S2": 30.0000000, "N2": 28.4397295, "K2": 30.0821373,
    "K1": 15.0410686, "O1": 13.9430356, "P1": 14.9589314, "Q1": 13.3986609,
    "M4": 57.9682084, "MS4": 58.9841042, "MN4": 57.4238337,
    "MF": 1.0980331, "MM": 0.5443747, "SSA": 0.0821373, "SA": 0.0410686,
}
TIDE_FIT_YEARS = 2
_TIDE_EPOCH = "2000-01-01T00:00"      # phase reference; any fixed instant works


def _hours_since_epoch(iso_local):
    import datetime as _dt
    a = _dt.datetime.fromisoformat(iso_local)
    b = _dt.datetime.fromisoformat(_TIDE_EPOCH)
    return (a - b).total_seconds() / 3600.0


@ttl_cache(ttl_seconds=30 * 86400, skip_none=True)
def fit_tide_harmonics():
    """Least-squares fit of tidal constituents to G-Land's own history.

    Returns {"mean": float, "terms": {name: [A, B]}, "rms_m": float} where the
    prediction is mean + sum(A*cos(wt) + B*sin(wt)). Cached for 30 days — the
    constituents are properties of the place, not of today.
    """
    import datetime as _dt
    try:
        import numpy as np
    except ImportError:
        print("[gland] numpy unavailable — harmonic tides disabled")
        return None

    end = _dt.date.today() - _dt.timedelta(days=7)   # stay clear of the forecast edge
    start = end - _dt.timedelta(days=365 * TIDE_FIT_YEARS)
    try:
        r = requests.get(MARINE_API, params={
            "latitude": GLAND_LAT, "longitude": GLAND_LON,
            "hourly": "sea_level_height_msl",
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": GLAND_TZ}, timeout=90)
        record_api_calls(1)
        if r.status_code != 200:
            print(f"[gland] harmonic fit HTTP {r.status_code}: {r.text[:160]}")
            return None
        h = r.json().get("hourly") or {}
    except Exception as e:
        print(f"[gland] harmonic fit {type(e).__name__}: {e}")
        return None

    times, vals = h.get("time") or [], h.get("sea_level_height_msl") or []
    t, y = [], []
    for ts, v in zip(times, vals):
        v = _safe(v)
        if v is None:
            continue
        t.append(_hours_since_epoch(ts))
        y.append(v)
    if len(y) < 24 * 400:
        print(f"[gland] harmonic fit: only {len(y)} samples, refusing")
        return None

    t = np.asarray(t); y = np.asarray(y)
    names = list(TIDE_CONSTITUENTS)
    cols = [np.ones_like(t)]
    for n in names:
        w = math.radians(TIDE_CONSTITUENTS[n])
        cols.append(np.cos(w * t)); cols.append(np.sin(w * t))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    out = {"mean": float(coef[0]), "terms": {}, "n": int(len(y)),
           "rms_m": float(np.sqrt(np.mean(resid ** 2)))}
    for i, n in enumerate(names):
        out["terms"][n] = [float(coef[1 + 2 * i]), float(coef[2 + 2 * i])]
    print(f"[gland] harmonic fit: {out['n']} samples, RMS {out['rms_m']*100:.1f} cm")
    return out


@ttl_cache(ttl_seconds=6 * 3600, skip_none=True)
def _tide_datum_offset():
    """Constant offset between the harmonic fit and the live tide field.

    The fit's mean is the mean of its 2-year training window, which sits a few
    cm off the datum the current forecast run uses (seasonal and interannual
    sea level that SA/SSA can't carry when extrapolated). Measured against the
    live field it is a near-constant ~0.5 ft, so anchoring removes it. Timing
    is unaffected — only the datum moves.
    """
    fit = fit_tide_harmonics()
    live = fetch_gland_tide()
    if not fit or not live:
        return 0.0
    try:
        import numpy as np
    except ImportError:
        return 0.0
    obs = {r["time"]: r["height_m"] for r in live.get("series", [])
           if r.get("height_m") is not None}
    if len(obs) < 24:
        return 0.0
    ts = sorted(obs)
    t = np.array([_hours_since_epoch(x) for x in ts])
    y = np.full(len(ts), fit["mean"])
    for name, (a, b) in fit["terms"].items():
        w = math.radians(TIDE_CONSTITUENTS[name])
        y += a * np.cos(w * t) + b * np.sin(w * t)
    off = float(np.mean(y - np.array([obs[x] for x in ts])))
    print(f"[gland] tide datum offset {off*100:+.1f} cm ({m_to_ft(off):+.2f} ft)")
    return off


def predict_tide_series(start_date: str, end_date: str):
    """Hourly harmonic tide prediction over an arbitrary date range."""
    import datetime as _dt
    fit = fit_tide_harmonics()
    if not fit:
        return None
    offset = _tide_datum_offset()
    try:
        import numpy as np
    except ImportError:
        return None
    d0 = _dt.datetime.fromisoformat(start_date + "T00:00")
    d1 = _dt.datetime.fromisoformat(end_date + "T23:00")
    n = int((d1 - d0).total_seconds() // 3600) + 1
    stamps = [d0 + _dt.timedelta(hours=i) for i in range(n)]
    t = np.array([_hours_since_epoch(s.isoformat(timespec="minutes")) for s in stamps])
    y = np.full(n, fit["mean"])
    for name, (a, b) in fit["terms"].items():
        w = math.radians(TIDE_CONSTITUENTS[name])
        y += a * np.cos(w * t) + b * np.sin(w * t)
    y = y - offset
    return [{"time": s.isoformat(timespec="minutes"),
             "height_m": float(v),
             "height_ft": round(m_to_ft(float(v)), 1)}
            for s, v in zip(stamps, y)]


@ttl_cache(ttl_seconds=7 * 86400, skip_none=True)
def fetch_gland_tide_range(start_date: str, end_date: str):
    """Tide highs/lows for an arbitrary date window, for the lookup tool.

    Driven by the harmonic fit rather than the forecast endpoint, so any date
    works — the forecast field only reaches ~9 days out. Cached hard for a
    week: for a given window the answer never changes.
    """
    series = predict_tide_series(start_date, end_date)
    if not series:
        return None
    events = _tide_events({"series": series})
    by_day = {}
    for e in events:
        by_day.setdefault(e["date"], []).append(e)

    import datetime as _dt
    days = []
    d, last = _dt.date.fromisoformat(start_date), _dt.date.fromisoformat(end_date)
    while d <= last:
        iso = d.isoformat()
        days.append({"date": iso, "events": by_day.get(iso, []), "no_data": False})
        d += _dt.timedelta(days=1)
    fit = fit_tide_harmonics() or {}
    return {"days": days, "start": start_date, "end": end_date,
            "source": "harmonic",
            "fit_rms_ft": round(m_to_ft(fit.get("rms_m", 0)), 2) if fit else None}


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_gland_wind():
    """Hourly wind at the point, in knots."""
    params = {
        "latitude": GLAND_LAT, "longitude": GLAND_LON,
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "forecast_days": FORECAST_DAYS,
        "timezone": GLAND_TZ,
        "wind_speed_unit": "kn",
    }
    try:
        r = requests.get(FORECAST_API, params=params, timeout=30)
        record_api_calls(1)
        if r.status_code != 200:
            return None
        h = (r.json().get("hourly") or {})
    except Exception as e:
        print(f"[gland] wind {type(e).__name__}: {e}")
        return None
    out = []
    for i, t in enumerate(h.get("time") or []):
        def g(k):
            arr = h.get(k)
            return _safe(arr[i]) if arr and i < len(arr) else None
        out.append({
            "time": t,
            "speed_kt": round(g("wind_speed_10m"), 1) if g("wind_speed_10m") is not None else None,
            "gust_kt": round(g("wind_gusts_10m"), 1) if g("wind_gusts_10m") is not None else None,
            "direction_deg": g("wind_direction_10m"),
        })
    return out


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_upstream_buoys():
    """Latest reading from each Western Australian sentinel buoy (AODN NRT).

    One WFS call sorted newest-first, then deduped to the most recent row per
    site. Rows carry Hs/Tp/Dp; some Spotter deployments publish Tp/Dp without
    Hs, which we surface as-is rather than inventing a value.
    """
    sites = ",".join(f"'{b['site']}'" for b in UPSTREAM_BUOYS)
    params = {
        "typeName": "aodn:aodn_wave_nrt_v2_timeseries_map",
        "SERVICE": "WFS", "REQUEST": "GetFeature", "VERSION": "1.1.0",
        "outputFormat": "application/json",
        "CQL_FILTER": f"site_name IN ({sites})",
        "sortBy": "TIME D",
        "maxFeatures": 400,
    }
    try:
        r = requests.get(AODN_WFS, params=params, timeout=45)
        record_api_calls(1)
        if r.status_code != 200:
            return None
        feats = (r.json() or {}).get("features") or []
    except Exception as e:
        print(f"[gland] AODN {type(e).__name__}: {e}")
        return None

    latest = {}
    for f in feats:
        p = f.get("properties") or {}
        name, t = p.get("site_name"), p.get("TIME")
        if not name or not t:
            continue
        if name not in latest or t > latest[name]["TIME"]:
            latest[name] = p

    out = []
    for b in UPSTREAM_BUOYS:
        p = latest.get(b["site"])
        dist = haversine_km(b["lat"], b["lon"], GLAND_LAT, GLAND_LON)
        brg = bearing_deg(b["lat"], b["lon"], GLAND_LAT, GLAND_LON)
        hs = _safe(p.get("significant_wave_height")) if p else None
        tp = _safe(p.get("peak_wave_period")) if p else None
        dp = _safe(p.get("peak_wave_direction")) if p else None
        eta = swell_travel_hours(dist, tp) if tp else None
        out.append({
            "site": b["site"],
            "lat": b["lat"], "lon": b["lon"],
            "distance_km": round(dist),
            "bearing_to_gland": round(brg),
            # Degrees off due north. 0 = G-Land sits straight up the meridian
            # from this buoy, so a swell passing it northbound is genuinely
            # heading at Java; larger = more off-axis, more sentinel than ray.
            "north_offset": round(abs(((brg + 180) % 360) - 180)),
            "time": p.get("TIME") if p else None,
            "hs_m": round(hs, 2) if hs is not None else None,
            "hs_ft": round(m_to_ft(hs), 1) if hs is not None else None,
            "tp_s": round(tp, 1) if tp is not None else None,
            "dp_deg": round(dp) if dp is not None else None,
            # Transit time for that period over that distance (cg = gT/4π).
            # A scale reference for how far ahead this buoy sits, NOT a
            # promise that this particular swell arrives at G-Land.
            "transit_hours": round(eta) if eta else None,
            "instrument": (p or {}).get("instrument"),
        })
    out.sort(key=lambda x: x["distance_km"])
    return out


# ── WA buoy → G-Land translation ────────────────────────────────────────────
# The method is the standard one from the swell-tracking literature (Collard,
# Ardhuin & Chapron 2009, "Monitoring and analysis of ocean swell fields from
# space"): a swell packet travels a great circle at the group speed of its peak
# period, cg = gT/4π, so you can back-project an observation to its source and
# forward-project that source anywhere else.
#
#   1. A buoy sees Hs/Tp/Dp at time t. The energy arrived on bearing Dp, so the
#      source lies along the great circle from the buoy on bearing Dp, at
#      distance cg·τ for some unknown travel time τ.
#   2. Sweep τ over a plausible range. Each τ gives a candidate source S.
#   3. From S, forward-project to G-Land: distance d, arrival bearing θ, and
#      arrival time t − τ + d/cg.
#   4. Keep only candidates where θ lands inside G-Land's swell window and the
#      arrival is still ahead of us.
#
# This does NOT require the buoy to sit on a G-Land ray — it almost never does.
# It requires only that the same storm radiates toward both, which is the actual
# physical relationship between the WA coast and Java.
#
# Height is scaled by great-circle geometric spreading. Energy from a point
# source falls off as 1/[α·sin α] with angular distance α, so amplitude scales
# as sqrt[(α_b·sin α_b)/(α_g·sin α_g)]. Dissipation is deliberately NOT modelled:
# for long-period swell it is small over these distances, and inventing a
# coefficient would add false precision to an already coarse estimate.

TRANSLATE_TAU_HOURS = (12, 192)    # plausible source travel times: 0.5–8 days
TRANSLATE_TAU_STEP = 3
_EARTH_R_KM = 6371.0


def _destination_point(lat, lon, bearing_deg, dist_km):
    """Great-circle destination from a point given bearing and distance."""
    p = math.pi / 180.0
    ang = dist_km / _EARTH_R_KM
    lat1, lon1, br = lat * p, lon * p, bearing_deg * p
    lat2 = math.asin(math.sin(lat1) * math.cos(ang)
                     + math.cos(lat1) * math.sin(ang) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(ang) * math.cos(lat1),
                             math.cos(ang) - math.sin(lat1) * math.sin(lat2))
    return lat2 / p, ((lon2 / p + 540) % 360) - 180


def _spread_factor(d_src_buoy_km, d_src_gland_km):
    """Amplitude ratio from great-circle geometric spreading alone."""
    a_b = max(1e-6, d_src_buoy_km / _EARTH_R_KM)
    a_g = max(1e-6, d_src_gland_km / _EARTH_R_KM)
    e_b = a_b * math.sin(a_b) if a_b < math.pi else 1e-6
    e_g = a_g * math.sin(a_g) if a_g < math.pi else 1e-6
    if e_g <= 0:
        return None
    return math.sqrt(max(0.0, e_b / e_g))


def translate_buoy(b, now_epoch):
    """Project one buoy reading forward to a G-Land arrival.

    Returns the best feasible solution (the one whose source best matches a
    typical Southern Ocean storm latitude) plus the feasible arrival window,
    or None when no candidate source puts G-Land inside the swell window.
    """
    tp, dp, hs = b.get("tp_s"), b.get("dp_deg"), b.get("hs_m")
    if tp is None or dp is None or not b.get("time"):
        return None
    try:
        import datetime as _dt
        t_obs = _dt.datetime.fromisoformat(
            b["time"].replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

    cg_kmh = 9.81 * tp / (4 * math.pi) * 3.6
    lo, hi = TRANSLATE_TAU_HOURS
    sols = []
    for tau in range(lo, hi + 1, TRANSLATE_TAU_STEP):
        d_sb = cg_kmh * tau
        s_lat, s_lon = _destination_point(b["lat"], b["lon"], dp, d_sb)
        if s_lat < -68 or s_lat > 5:            # off the planet's storm belt
            continue
        d_sg = haversine_km(s_lat, s_lon, GLAND_LAT, GLAND_LON)
        theta = bearing_deg(GLAND_LAT, GLAND_LON, s_lat, s_lon)
        if not (WINDOW_EDGE[0] <= theta <= WINDOW_EDGE[1]):
            continue
        arrive = t_obs - tau * 3600 + (d_sg / cg_kmh) * 3600
        lead_h = (arrive - now_epoch) / 3600.0
        if lead_h < -6 or lead_h > 216:          # already gone, or beyond range
            continue
        sf = _spread_factor(d_sb, d_sg)
        hs_g = round(m_to_ft(hs * sf), 1) if (hs is not None and sf) else None
        sols.append({
            "tau_h": tau,
            "source": [round(s_lat, 2), round(s_lon, 2)],
            "source_km": round(d_sg),
            "arrive_epoch": arrive,
            "lead_h": round(lead_h, 1),
            "arrive_dir": round(theta),
            "hs_ft": hs_g,
            "in_core": WINDOW_CORE[0] <= theta <= WINDOW_CORE[1],
        })
    if not sols:
        return None

    # Prefer sources in the Southern Ocean storm belt (Kerguelen-to-Heard is
    # where Indonesian swell is actually born) and inside the core window.
    def rank(s):
        lat = s["source"][0]
        belt = 0.0 if -60 <= lat <= -35 else min(abs(lat + 47.5) / 20.0, 2.0)
        return (0 if s["in_core"] else 1, belt)

    sols.sort(key=rank)
    best = sols[0]
    leads = [s["lead_h"] for s in sols]
    dirs = [s["arrive_dir"] for s in sols]
    return {
        "site": b["site"], "tp_s": tp, "dp_deg": dp,
        "hs_ft_buoy": b.get("hs_ft"), "hs_m": hs,
        "cg_kmh": round(cg_kmh),
        "best": best,
        "lead_range_h": [round(min(leads), 1), round(max(leads), 1)],
        "dir_range": [min(dirs), max(dirs)],
        "n_solutions": len(sols),
        # Private fields the triangulation needs; stripped before the payload.
        "_lat": b["lat"], "_lon": b["lon"],
        "_t_obs": t_obs, "_cg_kmh": cg_kmh,
    }


def _obs_epoch(b):
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(
            b["time"].replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _fit_source(members, lat_rng, lon_rng, step):
    """Grid-search the one storm position that best explains every buoy.

    Each buoy constrains the source two ways: the back-azimuth from the buoy
    must point at it, and the implied radiation time (t_obs − distance/cg) must
    agree with every other buoy's. A candidate is scored on both, so the fit is
    a genuine triangulation rather than a per-buoy guess averaged together.
    """
    best = None
    lat = lat_rng[0]
    while lat <= lat_rng[1]:
        lon = lon_rng[0]
        while lon <= lon_rng[1]:
            bearing_err, origins, ok = 0.0, [], True
            for m in members:
                d = haversine_km(m["_lat"], m["_lon"], lat, lon)
                if d < 400:                      # storm can't sit on the buoy
                    ok = False
                    break
                br = bearing_deg(m["_lat"], m["_lon"], lat, lon)
                err = abs(((br - m["dp_deg"] + 180) % 360) - 180)
                if err > 45:                     # ray simply doesn't point here
                    ok = False
                    break
                bearing_err += err
                origins.append(m["_t_obs"] - (d / m["_cg_kmh"]) * 3600)
            if ok and origins:
                mean_o = sum(origins) / len(origins)
                # Hours of disagreement about when the storm radiated.
                time_spread = (max(origins) - min(origins)) / 3600.0
                score = bearing_err / len(members) + time_spread * 1.5
                if best is None or score < best["score"]:
                    best = {"lat": lat, "lon": lon, "score": score,
                            "bearing_err": bearing_err / len(members),
                            "time_spread_h": time_spread, "origin": mean_o}
            lon += step
        lat += step
    return best


def _cluster_sources(translations, now_epoch):
    """Group buoys by peak period, then triangulate one source per group and
    forward-project it to G-Land."""
    groups = []
    for t in translations:
        placed = False
        for g in groups:
            if abs(g["tp_s"] - t["tp_s"]) <= 1.5:
                g["members"].append(t)
                g["tp_s"] = sum(m["tp_s"] for m in g["members"]) / len(g["members"])
                placed = True
                break
        if not placed:
            groups.append({"tp_s": t["tp_s"], "members": [t]})

    out = []
    for g in groups:
        ms = g["members"]
        coarse = _fit_source(ms, (-64.0, -24.0), (30.0, 132.0), 2.0)
        if not coarse:
            continue
        fit = _fit_source(ms, (coarse["lat"] - 2.5, coarse["lat"] + 2.5),
                          (coarse["lon"] - 2.5, coarse["lon"] + 2.5), 0.5) or coarse

        cg = 9.81 * g["tp_s"] / (4 * math.pi) * 3.6
        d_g = haversine_km(fit["lat"], fit["lon"], GLAND_LAT, GLAND_LON)
        theta = bearing_deg(GLAND_LAT, GLAND_LON, fit["lat"], fit["lon"])
        arrive = fit["origin"] + (d_g / cg) * 3600
        lead_h = (arrive - now_epoch) / 3600.0

        # Height: scale each buoy's Hs by geometric spreading from this source.
        hs_est = []
        for m in ms:
            if m["hs_m"] is None:
                continue
            d_b = haversine_km(m["_lat"], m["_lon"], fit["lat"], fit["lon"])
            sf = _spread_factor(d_b, d_g)
            if sf:
                hs_est.append(m["hs_m"] * sf)

        in_window = WINDOW_EDGE[0] <= theta <= WINDOW_EDGE[1]
        out.append({
            "tp_s": round(g["tp_s"], 1),
            "buoys": [m["site"] for m in ms],
            "n_buoys": len(ms),
            "source": [round(fit["lat"], 1), round(fit["lon"], 1)],
            "source_km": round(d_g),
            "bearing_err_deg": round(fit["bearing_err"], 1),
            "time_spread_h": round(fit["time_spread_h"], 1),
            "lead_h": round(lead_h, 1),
            "arrive_epoch": arrive,
            "arrive_dir": round(theta),
            "in_window": in_window,
            # Bands overlap, so a direction can feed more than one section.
            "band": " + ".join(dict.fromkeys(
                b["label"] for b in SWELL_BANDS
                if b["lo"] <= theta < b["hi"])) or None,
            "hs_ft": round(m_to_ft(sum(hs_est) / len(hs_est)), 1) if hs_est else None,
            # Confidence is about how well the rays actually converged, not how
            # many buoys happened to land in the group.
            "confidence": ("firm" if len(ms) >= 3 and fit["bearing_err"] < 8
                           and fit["time_spread_h"] < 12 else
                           "fair" if len(ms) >= 2 and fit["bearing_err"] < 15
                           and fit["time_spread_h"] < 24 else "weak"),
        })
    out.sort(key=lambda x: x["lead_h"])
    return out


def translate_upstream(buoys, now_epoch=None):
    """Full translation pass over the sentinel line."""
    if not buoys:
        return {"per_buoy": [], "clusters": []}
    now_epoch = now_epoch or time.time()
    per = [t for t in (translate_buoy(b, now_epoch) for b in buoys) if t]
    clusters = _cluster_sources(per, now_epoch)
    public = [{k: v for k, v in t.items() if not k.startswith("_")} for t in per]
    return {"per_buoy": public, "clusters": clusters}


# ── Buoy-derived vs model cross-check ───────────────────────────────────────
# The translation is an independent estimate: it comes from in-situ measurements
# and great-circle geometry, with no wave model anywhere in the chain. So
# comparing it against GFS and EURO at the hour it says the swell lands is a
# genuine second opinion, not a circular one. Direction is the axis to watch —
# it is the weakest output of the method and the one that decides which section
# of the reef a swell feeds.

# Thresholds for the verdict wording. Height in ft, direction in degrees.
_AGREE_HS, _CLOSE_HS = 1.0, 2.0
_AGREE_DIR, _CLOSE_DIR = 15, 30


def _verdict(d_hs, d_dir):
    if d_hs is None and d_dir is None:
        return None
    hs_ok = d_hs is None or abs(d_hs) <= _AGREE_HS
    dir_ok = d_dir is None or abs(d_dir) <= _AGREE_DIR
    if hs_ok and dir_ok:
        return "agree"
    hs_cl = d_hs is None or abs(d_hs) <= _CLOSE_HS
    dir_cl = d_dir is None or abs(d_dir) <= _CLOSE_DIR
    if hs_cl and dir_cl:
        return "close"
    return "diverge"


def _angle_delta(a, b):
    if a is None or b is None:
        return None
    return round(((a - b + 180) % 360) - 180)


def compare_translation_to_models(clusters, timeline):
    """Annotate each in-window cluster with what the models say at the hour it
    predicts the swell arrives."""
    if not clusters or not timeline:
        return clusters
    by_time = {r["time"]: r for r in timeline}
    times = sorted(by_time)

    for c in clusters:
        if not c.get("in_window") or not c.get("arrive_epoch"):
            continue
        # Timeline keys are local (WIB = UTC+7), rounded to the nearest hour.
        local = time.gmtime(c["arrive_epoch"] + 7 * 3600 + 1800)
        key = time.strftime("%Y-%m-%dT%H", local)
        if key not in by_time:
            future = [t for t in times if t >= key]
            key = future[0] if future else (times[-1] if times else None)
        row = by_time.get(key)
        if not row:
            continue

        check = {"time": key, "models": {}}
        for label, field in (("GFS", "gfs"), ("EURO", "euro")):
            sw = row.get(field)
            if not sw:
                check["models"][label] = {"in_window": False}
                continue
            d_hs = (round(c["hs_ft"] - sw["height_ft"], 1)
                    if c.get("hs_ft") is not None else None)
            d_dir = _angle_delta(c.get("arrive_dir"), sw.get("direction_deg"))
            d_tp = (round(c["tp_s"] - sw["period_s"], 1)
                    if sw.get("period_s") is not None else None)
            check["models"][label] = {
                "in_window": True,
                "hs_ft": sw["height_ft"],
                "period_s": round(sw["period_s"], 1) if sw.get("period_s") else None,
                "direction_deg": round(sw["direction_deg"]) if sw.get("direction_deg") is not None else None,
                "d_hs": d_hs, "d_dir": d_dir, "d_tp": d_tp,
                "verdict": _verdict(d_hs, d_dir),
            }
        verdicts = [m.get("verdict") for m in check["models"].values() if m.get("verdict")]
        check["best"] = ("agree" if "agree" in verdicts else
                         "close" if "close" in verdicts else
                         "diverge" if verdicts else None)
        c["model_check"] = check
    return clusters


# ── Section ranking ─────────────────────────────────────────────────────────
def _quality_ceiling(face_ft, period_s):
    """How good can *any* section plausibly be right now, in absolute terms?

    Without this the relative ranking lies: in 3 ft of swell the softest
    section still "fits" perfectly and scores in the 80s, which reads as
    epic. This caps the whole reef by how much swell is actually in the
    water, so a relative winner in marginal surf still shows as marginal.
    """
    if face_ft is None:
        return 0.0
    if face_ft < 2.5:
        ceil = 0.18
    elif face_ft < 4:
        ceil = 0.45
    elif face_ft < 6:
        ceil = 0.70
    elif face_ft < 8:
        ceil = 0.88
    else:
        ceil = 1.0
    # Long-period energy is worth more than the same height of short-period.
    if period_s:
        if period_s >= 15:
            ceil = min(1.0, ceil * 1.15)
        elif period_s < 11:
            ceil *= 0.88
    return ceil


def rank_sections(swell, wind_speed_kt, wind_dir_deg, tide_ft, tide_state):
    """Score every section of the reef for one hour's conditions.

    Deliberately not a single "surf score" — G-Land is not one wave. The same
    hour can be maxing and unsurfable at Speedies while Kongs is perfect, so
    the page ranks the reef instead of rating the spot.
    """
    out = []
    face_ft = None
    if swell and swell.get("height_ft"):
        face_ft = swell["height_ft"]
    period_s = swell.get("period_s") if swell else None
    ceiling = _quality_ceiling(face_ft, period_s)

    for sec in SECTIONS:
        reasons = []
        score = 0.0

        # ── Size fit: trapezoid over this section's own working range ──
        if face_ft is None:
            size_fit = 0.0
            reasons.append("no swell in window")
        else:
            lo_min = sec["min_face_ft"]
            id_lo, id_hi = sec["ideal_face_ft"]
            hi_max = sec["size_ft"][1] * HAWAIIAN_TO_FACE
            if face_ft < lo_min:
                size_fit = 0.0
                reasons.append(f"under {lo_min} ft — not working")
            elif face_ft < id_lo:
                size_fit = (face_ft - lo_min) / max(1e-6, id_lo - lo_min)
                reasons.append("on the small side for this section")
            elif face_ft <= id_hi:
                size_fit = 1.0
            elif face_ft <= hi_max:
                span = max(1e-6, hi_max - id_hi)
                size_fit = 1.0 - 0.6 * (face_ft - id_hi) / span
            else:
                size_fit = 0.3
                reasons.append("over the top of its range")
        # Size is a GATE, not a contribution. A section that is not working
        # cannot accumulate points from good wind and a nice tide — Speedies
        # on a 3 ft day is zero, not "a bit off".

        # ── Direction fit ──
        dir_fit = 0.5
        if swell and swell.get("direction_deg") is not None:
            d = swell["direction_deg"]
            lo, hi = sec["best_dir"]
            if lo <= d <= hi:
                dir_fit = 1.0
            else:
                miss = min(abs(d - lo), abs(d - hi))
                dir_fit = max(0.0, 1.0 - miss / 45.0)
            if dir_fit < 0.4:
                reasons.append("swell angle off for this section")
        score += dir_fit * 30

        # ── Period bonus: the growers ──
        if swell and swell.get("period_s"):
            p = swell["period_s"]
            if sec["key"] in ("speedies", "moneytrees") and p >= 14:
                score += 15
                reasons.append(f"{p:.0f} s lines will wrap and grow")
            elif p >= 14:
                score += 11
            elif p >= 12:
                score += 7

        # ── Tide ──
        tide_fit = 1.0
        if tide_ft is not None:
            want = sec["tide"]
            if want == "high":
                tide_fit = 0.15 if tide_state == "low" else (
                    0.6 if tide_state == "mid" else 1.0)
                if tide_state == "low":
                    reasons.append("needs water — dry reef risk")
            elif want == "mid-high":
                tide_fit = 0.45 if tide_state == "low" else 1.0
            elif want == "mid":
                tide_fit = 0.7 if tide_state in ("low", "high") else 1.0
            else:
                tide_fit = 1.0
        score += tide_fit * 30

        # ── Wind ──
        w = wind_for_section(sec, wind_speed_kt, wind_dir_deg)
        wind_fit = {
            "GLASSY": 1.0, "GROOMED": 1.0, "CLEAN": 0.8,
            "STRONG OFFSHORE": 0.75, "TEXTURED": 0.5,
            "CHOPPY": 0.2, "ONSHORE": 0.05,
        }.get(w["rating"], 0.5)
        score += wind_fit * 25
        if wind_fit <= 0.2:
            reasons.append("wind is wrong for this stretch of reef")

        # `prestige` is the intrinsic quality of that piece of reef, so a
        # working Moneytrees outranks a working Chickens instead of losing to
        # it just because Chickens tolerates smaller surf. Surfline is explicit
        # that Kongs is "not usually a barrel, nor genuinely world-class" and
        # that Chickens is "a slightly lame little left end section", while
        # Moneytrees and Speed Reef are the world-class waves.
        out.append({
            "key": sec["key"], "name": sec["name"],
            "score": round(min(100.0, score) * size_fit * ceiling * sec["prestige"]),
            "raw_score": round(min(100.0, score)),
            "size_fit": round(size_fit, 2),
            "ceiling": round(ceiling, 2),
            "prestige": sec["prestige"],
            "wind": w,
            "tide_fit": round(tide_fit, 2),
            "reasons": reasons,
            "blurb": sec["blurb"],
            "detail": sec["detail"],
            "tide_note": sec["tide_note"],
            "offshore": sec["offshore"],
            "faces": sec["faces"],
            "size_ft": sec["size_ft"],
        })
    out.sort(key=lambda s: s["score"], reverse=True)
    return out


def tide_state_for(height_ft, lo_ft, hi_ft):
    """Bucket an hour's tide height within the day's own range."""
    if height_ft is None or lo_ft is None or hi_ft is None or hi_ft <= lo_ft:
        return None
    pct = (height_ft - lo_ft) / (hi_ft - lo_ft)
    if pct < 0.33:
        return "low"
    if pct < 0.67:
        return "mid"
    return "high"


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fun_plus_summary():
    """G-Land's Fun+ Days figure for the main dashboard's overview column.

    Mirrors index.html's computeModelOverview criteria exactly: sample both
    models on a 3 h stride from now, skip night, take min(EURO, GFS) category,
    count a window when that minimum is FUN or better AND the wind is
    surfable, then count days holding >= 2 such windows. Denominator is the
    span of sampled times in days.

    Two things are G-Land's rather than the site's, because they are what the
    /gland page itself uses: the category scheme (its own TOML) and the wind
    gate (the point's own offshore bearing, not a region of spots).
    """
    data = fetch_all()
    timeline = data.get("timeline") or []
    sun = data.get("sun") or {}
    if not timeline:
        return None

    now_key = time.strftime("%Y-%m-%dT%H",
                            time.gmtime(time.time() + _tz_offset_seconds()))
    rows = [r for r in timeline if r["time"] >= now_key]
    if len(rows) < 2:
        return None
    base = int(rows[0]["time"][11:13])
    sampled = [r for r in rows if (int(r["time"][11:13]) - base) % 3 == 0]

    order = {c: i for i, c in enumerate(GLAND_CATEGORIES)}
    fun_i = order.get("FUN", 2)
    surfable = {"GLASSY", "GROOMED", "CLEAN", "TEXTURED", "STRONG OFFSHORE"}

    per_day, best_i, best = {}, -1, None
    for r in sampled:
        day, hh = r["time"][:10], int(r["time"][11:13])
        # Daylight from the location's own sunrise/sunset rather than a fixed
        # window — G-Land is 8.7°S and its day is ~11.7 h year-round.
        s = sun.get(day)
        if s:
            if not (int(s["sunrise"][11:13]) <= hh < int(s["sunset"][11:13])):
                continue
        elif hh < 6 or hh >= 18:
            continue

        g, e = r.get("gfs"), r.get("euro")
        if not g or not e:
            continue
        gc = categorize_gland(g["height_ft"], g["period_s"], g.get("direction_deg"))
        ec = categorize_gland(e["height_ft"], e["period_s"], e.get("direction_deg"))
        gi, ei = order.get(gc, -1), order.get(ec, -1)
        if gi < 0 or ei < 0:
            continue
        m = min(gi, ei)
        if m > best_i:
            best_i, best = m, (gc if gi <= ei else ec)
        if m < fun_i:
            continue
        if r.get("wind_rating") and r["wind_rating"] not in surfable:
            continue
        per_day[day] = per_day.get(day, 0) + 1

    count = sum(1 for v in per_day.values() if v >= 2)
    import datetime as _dt
    d0 = _dt.datetime.fromisoformat(sampled[0]["time"] + ":00")
    d1 = _dt.datetime.fromisoformat(sampled[-1]["time"] + ":00")
    window_days = round((d1 - d0).total_seconds() / 86400)
    cat = best or "FLAT"
    col = GLAND_COLORS.get(cat, GLAND_COLORS["FLAT"])
    return {"count": count, "window_days": window_days, "category": cat,
            "colors": col, "url": "/gland", "name": "G-LAND"}


HISTORY_DAYS = 14


@ttl_cache(ttl_seconds=3600, skip_none=True)
def fetch_gland_history(days: int = HISTORY_DAYS):
    """The last `days` of observed-ish conditions at G-Land.

    Nothing is archived locally — Open-Meteo serves its own past analysis for
    both the wave model and wind, and tide comes from the harmonic fit, so a
    look-back is just three calls. Rows come back in the SAME shape as the
    forecast timeline so the table renderer needs no special case.

    ECMWF-WAM comes from the rolling local archive (gland_euro_archive), not
    from a live call: waves_cmems is pinned to the forecast window and CMEMS
    is rate-capped, so past EURO has to be persisted as it goes by.
    """
    gfs = _fetch_marine_past(days)
    wind = _fetch_wind_past(days)
    if not gfs and not wind:
        return None

    euro, euro_status = None, {"rows": 0}
    try:
        import gland_euro_archive as _arch
        euro_status = _arch.archive_status()
        cutoff = (gfs or wind)[0]["time"][:13]
        euro = [r for r in _arch.load_archive_rows()
                if r["time"][:13] >= cutoff] or None
    except Exception as e:
        print(f"[gland] euro archive {type(e).__name__}: {e}")

    tide_series = None
    if gfs or wind:
        ref = gfs or wind
        tide_series = predict_tide_series(ref[0]["time"][:10], ref[-1]["time"][:10])
    tide = {"series": tide_series or []}
    if tide_series:
        hilo = []
        for i in range(1, len(tide_series) - 1):
            a, b, c = (tide_series[i - 1]["height_m"], tide_series[i]["height_m"],
                       tide_series[i + 1]["height_m"])
            if b >= a and b >= c and not (a == b == c):
                hilo.append({"time": tide_series[i]["time"], "type": "H",
                             "height_ft": tide_series[i]["height_ft"]})
            elif b <= a and b <= c and not (a == b == c):
                hilo.append({"time": tide_series[i]["time"], "type": "L",
                             "height_ft": tide_series[i]["height_ft"]})
        tide["hilo"] = hilo

    timeline = _build_timeline(gfs, euro, wind, tide)
    return {
        "timeline": timeline,
        "days": days,
        "has_euro": bool(euro),
        "euro_archive": euro_status,
        "tide_events": _tide_events(tide),
        "sun": _sun_range(timeline[0]["time"][:10], timeline[-1]["time"][:10])
               if timeline else {},
    }


def _fetch_marine_past(days: int):
    """GFS-Wave partitions for the past `days`, at the offshore node."""
    for model_id in _GFS_MODEL_IDS:
        try:
            r = requests.get(MARINE_API, params={
                "latitude": SWELL_NODE_LAT, "longitude": SWELL_NODE_LON,
                "hourly": ",".join(_WAVE_VARS), "models": model_id,
                "past_days": days, "forecast_days": 1, "timezone": GLAND_TZ,
            }, timeout=45)
            record_api_calls(1)
            if r.status_code != 200:
                continue
            rows = _parse_marine(r.json(), "GFS")
            # Drop anything at or after the current hour — that's the forecast's
            # job. gmtime + the WIB offset, NOT localtime: this box runs on ET.
            cut = time.strftime("%Y-%m-%dT%H",
                                time.gmtime(time.time() + _tz_offset_seconds()))
            return [x for x in rows if x["time"][:13] < cut]
        except Exception as e:
            print(f"[gland] history GFS {model_id} {type(e).__name__}: {e}")
    return None


def _fetch_wind_past(days: int):
    """Wind at the point for the past `days`."""
    try:
        r = requests.get(FORECAST_API, params={
            "latitude": GLAND_LAT, "longitude": GLAND_LON,
            "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "past_days": days, "forecast_days": 1, "timezone": GLAND_TZ,
            "wind_speed_unit": "kn",
        }, timeout=45)
        record_api_calls(1)
        if r.status_code != 200:
            return None
        h = r.json().get("hourly") or {}
    except Exception as e:
        print(f"[gland] history wind {type(e).__name__}: {e}")
        return None
    out = []
    for i, t in enumerate(h.get("time") or []):
        def g(k):
            a = h.get(k)
            return _safe(a[i]) if a and i < len(a) else None
        out.append({"time": t,
                    "speed_kt": round(g("wind_speed_10m"), 1) if g("wind_speed_10m") is not None else None,
                    "gust_kt": round(g("wind_gusts_10m"), 1) if g("wind_gusts_10m") is not None else None,
                    "direction_deg": g("wind_direction_10m")})
    return out


def _tz_offset_seconds():
    """WIB is UTC+7 year-round (no DST), so this is a constant."""
    return 7 * 3600


def _sun_range(start_date: str, end_date: str):
    """Sunrise/sunset over an arbitrary past range (sun.compute_sun_data only
    counts forward from today)."""
    import datetime as _dt
    try:
        from astral import LocationInfo
        from astral.sun import sun as _sun
        from zoneinfo import ZoneInfo
    except ImportError:
        return {}
    tz = ZoneInfo(GLAND_TZ)
    loc = LocationInfo(latitude=GLAND_LAT, longitude=GLAND_LON, timezone=GLAND_TZ)
    out = {}
    d = _dt.date.fromisoformat(start_date)
    last = _dt.date.fromisoformat(end_date)
    while d <= last:
        try:
            s = _sun(loc.observer, date=d, tzinfo=tz)
            out[d.isoformat()] = {"sunrise": s["sunrise"].strftime("%Y-%m-%dT%H:%M"),
                                  "sunset": s["sunset"].strftime("%Y-%m-%dT%H:%M")}
        except Exception:
            pass
        d += _dt.timedelta(days=1)
    return out


def fill_tide_gaps(tide):
    """Extend the tide series across the whole forecast window.

    Open-Meteo's tide field stops ~9 days out while the wave models run 10, so
    the last rows of the table had no tide at all. The harmonic fit covers any
    date, so it backfills the tail — and, being datum-anchored to this same
    series, joins it without a step.
    """
    if not tide or not tide.get("series"):
        return tide
    series = tide["series"]
    missing = [r for r in series if r.get("height_ft") is None]
    if not missing:
        return tide
    pred = predict_tide_series(series[0]["time"][:10], series[-1]["time"][:10])
    if not pred:
        return tide
    by_time = {r["time"]: r for r in pred}
    filled = 0
    for r in series:
        if r.get("height_ft") is None and r["time"] in by_time:
            p = by_time[r["time"]]
            r["height_m"], r["height_ft"] = p["height_m"], p["height_ft"]
            r["modelled"] = True
            filled += 1
    if filled:
        print(f"[gland] tide: filled {filled} h from the harmonic model")
        # Rebuild hilo in the SAME shape fetch_gland_tide emits (full ISO
        # timestamps). _tide_events returns split date/clock fields for the
        # lookup tool, and feeding those to _build_timeline invents rows.
        hilo = []
        for i in range(1, len(series) - 1):
            a, b, c = (series[i - 1]["height_m"], series[i]["height_m"],
                       series[i + 1]["height_m"])
            if a is None or b is None or c is None:
                continue
            if b >= a and b >= c and not (a == b == c):
                hilo.append({"time": series[i]["time"], "type": "H",
                             "height_ft": series[i]["height_ft"]})
            elif b <= a and b <= c and not (a == b == c):
                hilo.append({"time": series[i]["time"], "type": "L",
                             "height_ft": series[i]["height_ft"]})
        tide["hilo"] = hilo
        heights = [x["height_m"] for x in series if x["height_m"] is not None]
        if heights:
            tide["range_m"] = round(max(heights) - min(heights), 2)
    return tide


def _tide_events(tide):
    """High/low turning points with their real clock time, refined off the
    hourly series by fitting a parabola through the extremum and its two
    neighbours — the true turn rarely lands exactly on the hour."""
    series = (tide or {}).get("series") or []
    if len(series) < 3:
        return []
    out = []
    for i in range(1, len(series) - 1):
        a, b, c = (series[i - 1].get("height_ft"), series[i].get("height_ft"),
                   series[i + 1].get("height_ft"))
        if a is None or b is None or c is None:
            continue
        is_hi = b >= a and b >= c and not (a == b == c)
        is_lo = b <= a and b <= c and not (a == b == c)
        if not (is_hi or is_lo):
            continue
        denom = (a - 2 * b + c)
        # Vertex offset in hours, clamped — a flat denominator means a plateau.
        off = 0.0 if abs(denom) < 1e-9 else max(-0.5, min(0.5, 0.5 * (a - c) / denom))
        peak = b if abs(denom) < 1e-9 else b - 0.25 * (a - c) * off
        t = series[i]["time"]                    # local 'YYYY-MM-DDTHH:MM'
        hh = int(t[11:13])
        mins = int(round((hh + off) * 60))
        day, adj = t[:10], 0
        if mins < 0:
            mins += 1440; adj = -1
        elif mins >= 1440:
            mins -= 1440; adj = 1
        if adj:
            import datetime as _dt
            day = (_dt.date.fromisoformat(day) + _dt.timedelta(days=adj)).isoformat()
        ev = {"date": day, "time": f"{mins // 60:02d}:{mins % 60:02d}",
              "type": "H" if is_hi else "L", "height_ft": round(peak, 1),
              "_i": i}
        # A flat top spans two hours, so b>=a and b>=c fires on both — keep the
        # more extreme of the pair rather than emitting the turn twice.
        if out and out[-1]["type"] == ev["type"] and i - out[-1]["_i"] <= 3:
            better = (ev["height_ft"] > out[-1]["height_ft"]) if is_hi \
                else (ev["height_ft"] < out[-1]["height_ft"])
            if better:
                out[-1] = ev
            continue
        out.append(ev)
    for e in out:
        del e["_i"]
    return out


def _build_timeline(gfs, euro, wind, tide):
    """Merge every source onto one hourly timeline and score the reef at each
    step, for both models.

    Section scoring lives here rather than in the template so there is one
    implementation of it. The page renders these numbers; it does not
    recompute them.
    """
    rows = {}

    def slot(t):
        if not t:
            return None
        key = t[:13]                     # YYYY-MM-DDTHH
        rows.setdefault(key, {"time": key})
        return rows[key]

    for r in gfs or []:
        s = slot(r["time"])
        if s is not None:
            s["gfs"] = r.get("gland_swell")
            s["gfs_all"] = r.get("components")
    for r in euro or []:
        s = slot(r["time"])
        if s is not None:
            s["euro"] = r.get("gland_swell")
            s["euro_all"] = r.get("components")
    for r in wind or []:
        s = slot(r["time"])
        if s is not None:
            s["wind"] = {"speed_kt": r.get("speed_kt"),
                         "gust_kt": r.get("gust_kt"),
                         "direction_deg": r.get("direction_deg")}
    for r in (tide or {}).get("series", []):
        s = slot(r["time"])
        if s is not None:
            s["tide_ft"] = r.get("height_ft")
    for r in (tide or {}).get("hilo", []):
        s = slot(r["time"])
        if s is not None:
            s["hilo"] = r.get("type")

    timeline = [rows[k] for k in sorted(rows)]

    # Per-day tide range, so low/mid/high is relative to that day's own swing.
    day_range = {}
    for r in timeline:
        ft = r.get("tide_ft")
        if ft is None:
            continue
        d = r["time"][:10]
        lo, hi = day_range.get(d, (ft, ft))
        day_range[d] = (min(lo, ft), max(hi, ft))

    keys = [s["key"] for s in SECTIONS]
    for r in timeline:
        d = r["time"][:10]
        rng = day_range.get(d)
        r["tide_state"] = tide_state_for(r.get("tide_ft"),
                                         rng[0] if rng else None,
                                         rng[1] if rng else None)
        w = r.get("wind") or {}
        for model in ("gfs", "euro"):
            ranked = rank_sections(r.get(model), w.get("speed_kt"),
                                   w.get("direction_deg"),
                                   r.get("tide_ft"), r.get("tide_state"))
            by_key = {s["key"]: s for s in ranked}
            # Compact: scores in SECTIONS order, plus the winning section.
            r[f"sec_{model}"] = [by_key[k]["score"] for k in keys]
            r[f"best_{model}"] = ranked[0]["key"] if ranked else None
        # One representative wind rating (Moneytrees, the middle of the reef).
        r["wind_rating"] = wind_for_section(
            SECTIONS[1], w.get("speed_kt"), w.get("direction_deg"))["rating"]
    return timeline


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_upstream_model_swell():
    """GFS and EURO sea state at each sentinel buoy, for the map overlay.

    Deliberately the COMBINED field (wave_height / wave_period / wave_direction),
    not swell partitions: a waverider reports significant wave height, peak
    period and peak direction for the whole sea state, so combined is the
    like-for-like comparison against an observation. This is the one place on
    the page that uses combined values, and it is model-vs-buoy — the G-Land
    forecast table stays on primary swell, where model-vs-model lives.
    """
    lats = ",".join(str(b["lat"]) for b in UPSTREAM_BUOYS)
    lons = ",".join(str(b["lon"]) for b in UPSTREAM_BUOYS)
    out = {}
    for label, model in (("GFS", "ncep_gfswave025"), ("EURO", "ecmwf_wam025")):
        try:
            r = requests.get(MARINE_API, params={
                "latitude": lats, "longitude": lons,
                "hourly": "wave_height,wave_period,wave_direction",
                "models": model, "forecast_days": 1, "timezone": "UTC",
            }, timeout=30)
            record_api_calls(1)
            if r.status_code != 200:
                continue
            payload = r.json()
            if isinstance(payload, dict):
                payload = [payload]
        except Exception as e:
            print(f"[gland] upstream model {label} {type(e).__name__}: {e}")
            continue

        now = time.gmtime()
        for b, d in zip(UPSTREAM_BUOYS, payload):
            h = d.get("hourly") or {}
            times = h.get("time") or []
            stamp = time.strftime("%Y-%m-%dT%H:00", now)
            i = times.index(stamp) if stamp in times else min(len(times) - 1,
                                                              max(0, now.tm_hour))
            if i < 0:
                continue

            def g(k):
                a = h.get(k)
                return _safe(a[i]) if a and i < len(a) else None

            hm, pp, dd = g("wave_height"), g("wave_period"), g("wave_direction")
            if hm is None:
                continue
            out.setdefault(b["site"], {})[label] = {
                "hs_ft": round(m_to_ft(hm), 1),
                "tp_s": round(pp, 1) if pp is not None else None,
                "dp_deg": round(dd) if dd is not None else None,
            }
    return out or None


def fetch_all():
    """Everything the /gland page needs, fetched in parallel."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {
            "gfs": pool.submit(fetch_gfs_waves),
            "euro": pool.submit(fetch_euro_waves),
            "tide": pool.submit(fetch_gland_tide),
            "wind": pool.submit(fetch_gland_wind),
            "upstream": pool.submit(fetch_upstream_buoys),
            "upstream_models": pool.submit(fetch_upstream_model_swell),
        }
        raw = {}
        for k, f in jobs.items():
            try:
                raw[k] = f.result()
            except Exception as e:
                print(f"[gland] {k} failed: {type(e).__name__}: {e}")
                raw[k] = None

    raw["tide"] = fill_tide_gaps(raw["tide"])
    timeline = _build_timeline(raw["gfs"], raw["euro"], raw["wind"], raw["tide"])

    # Sunrise/sunset at the point, and the tide turning points at their real
    # clock times — the timeline is hourly (and rendered on a 3 h stride), so
    # the exact minute of a high or low would otherwise be lost.
    try:
        from sun import compute_sun_data
        sun_data = compute_sun_data(GLAND_LAT, GLAND_LON, GLAND_TZ,
                                    days=FORECAST_DAYS)
    except Exception as e:
        print(f"[gland] sun {type(e).__name__}: {e}")
        sun_data = {}
    hilo_events = _tide_events(raw["tide"])
    translation = translate_upstream(raw["upstream"])
    compare_translation_to_models(translation["clusters"], timeline)

    ph = moon_phase(time.time())
    return {
        "timeline": timeline,
        "upstream": raw["upstream"],
        "upstream_models": raw.get("upstream_models"),
        "translation": translation,
        "tide_range_m": (raw["tide"] or {}).get("range_m"),
        "tide_source": (raw["tide"] or {}).get("source"),
        "sun": sun_data,
        "tide_events": hilo_events,
        "moon": {
            "phase": round(ph, 3),
            "label": moon_label(ph),
            "spring": is_spring_tide(ph),
        },
        "meta": {
            "lat": GLAND_LAT, "lon": GLAND_LON, "tz": GLAND_TZ,
            "have_gfs": bool(raw["gfs"]), "have_euro": bool(raw["euro"]),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
