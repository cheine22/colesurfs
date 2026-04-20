"""Shared constants for the CSC package — buoy list, paths, schemas."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSC_DATA_DIR = PROJECT_ROOT / ".csc_data"
CSC_MODELS_DIR = PROJECT_ROOT / ".csc_models"

FORECASTS_DIR = CSC_DATA_DIR / "forecasts"
OBSERVATIONS_DIR = CSC_DATA_DIR / "observations"
PREDICTIONS_DIR = CSC_DATA_DIR / "predictions"
LIVE_LOG_DIR = CSC_DATA_DIR / "live_log"
AUDIT_DIR = CSC_DATA_DIR / "audit"

# Archive start — Open-Meteo historical-forecast coverage is reliable
# from mid-2022 onward.
ARCHIVE_START = "2022-06-01"

# ─── CSC-scope buoys ──────────────────────────────────────────────────────
# (buoy_id, label, lat, lon, operator) — operator picks the historical
# archive: CDIP for 46221/46268, NDBC for everything else.
#
# NOTE: 44258 (Halifax) was originally in scope but has no NDBC coverage
# (ECCC-operated, not re-streamed). Dropped from CSC.
#
# SCOPE SEPARATION (v3 onward):
#   Training, evaluation, and display pipelines are bifurcated into two
#   independent tracks — `east` and `west` — that share no training data,
#   model artifacts, or dashboard surface. East is the user-facing track.
#   West trains silently in the background on a parallel launchd plist.
#
#   EAST_BUOYS requires ≥2 years of historical spectral data per buoy.
#   Barnegat (44091) and Jeffrey's Ledge (44098) were deployed in mid-2025
#   and do not yet meet that threshold — they'll be promoted to
#   EAST_BUOYS automatically once their archives pass 24 months.
BUOYS = [
    ("44065", "NY Harbor Entrance",    40.369, -73.703, "ndbc"),
    ("44097", "Block Island Sound",    40.969, -71.127, "ndbc"),
    ("44013", "Boston (MA)",           42.346, -70.651, "ndbc"),
    ("44098", "Jeffrey's Ledge (NH)",  42.798, -70.168, "ndbc"),
    ("44091", "Barnegat (NJ)",         39.778, -73.770, "ndbc"),
    ("46221", "Santa Monica Bay",      33.855, -118.641, "cdip"),
    ("46268", "Topanga Nearshore",     34.032, -118.641, "cdip"),
    ("46025", "Santa Monica Basin",    33.755, -119.045, "ndbc"),
]

BUOY_IDS = [b[0] for b in BUOYS]

# ─── Scope definitions (v3) ───────────────────────────────────────────────
EAST_BUOYS = ["44013", "44065", "44097"]    # Boston, NY Harbor, Block Island
WEST_BUOYS = ["46025", "46221", "46268"]    # Santa Monica Basin/Bay, Topanga

# Buoys deployed too recently to satisfy the ≥2-year training-data rule.
# These are still part of the CSC universe (the live logger captures them)
# but are EXCLUDED from East Coast training / display until their NDBC
# THREDDS archive accumulates 24 months. When a buoy crosses that
# threshold, move its id from FUTURE_EAST_BUOYS into EAST_BUOYS and re-run
# the East bakeoff.
FUTURE_EAST_BUOYS = {
    "44091": {"deployed": "2025-06-25", "eligible": "2027-06-25"},
    "44098": {"deployed": "2025-08-19", "eligible": "2027-08-19"},
}

DEFAULT_SCOPE = "east"   # user-facing dashboard scope


def buoys_for_scope(scope: str) -> list[str]:
    """Return the ordered buoy_id list for a named scope."""
    if scope == "east":
        return list(EAST_BUOYS)
    if scope == "west":
        return list(WEST_BUOYS)
    raise ValueError(f"unknown CSC scope: {scope!r} (expected 'east' or 'west')")


def scope_of(buoy_id: str) -> str:
    """Return 'east' or 'west' for any known CSC buoy; raises otherwise."""
    if buoy_id in EAST_BUOYS:
        return "east"
    if buoy_id in WEST_BUOYS:
        return "west"
    if buoy_id in FUTURE_EAST_BUOYS:
        return "east"    # logically east even while excluded from training
    raise ValueError(f"buoy_id {buoy_id!r} not in any CSC scope")


def models_dir_for_scope(scope: str):
    """Per-scope model-artifact root. East uses the existing `.csc_models/`
    (since v3 supersedes v2 there); West gets its own sibling directory so
    nothing from West ever leaks onto the dashboard."""
    if scope == "east":
        return CSC_MODELS_DIR
    if scope == "west":
        return CSC_MODELS_DIR.with_name(".csc_models_west")
    raise ValueError(f"unknown scope: {scope!r}")

# Open-Meteo model identifiers (Marine API `models=` parameter)
OM_MODELS = {
    "GFS":  "ncep_gfswave025",
    "EURO": "ecmwf_wam025",
}

# Model-run / forecast cycles issued per day (UTC). Used for expected-cycle
# accounting in the audit and backfill.
CYCLES_PER_DAY = {
    "GFS":  [0, 6, 12, 18],
    "EURO": [0, 12],
}

# Marine-API "bulk" variables per query. Combined (wave_*) + three swell
# partitions. The logger requests these plus their `*_previous_day1..7`
# companions to build a lead-time-indexed archive going forward.
OM_WAVE_VARS = [
    "wave_height", "wave_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "secondary_swell_wave_height", "secondary_swell_wave_period", "secondary_swell_wave_direction",
]

# The `_previous_dayN` rolling window — 1..7 d back from today, matching
# Open-Meteo's exposed lookback. Represents lead times 24/48/.../168 h.
PREV_DAY_LEADS = [1, 2, 3, 4, 5, 6, 7]

def om_var_columns() -> list[str]:
    """All Marine API variable column names the logger requests, including
    the previous_day lookback companions."""
    cols = list(OM_WAVE_VARS)
    for v in OM_WAVE_VARS:
        for n in PREV_DAY_LEADS:
            cols.append(f"{v}_previous_day{n}")
    return cols

# Lead-time buckets used for reporting and stratification.
LEAD_BUCKETS = [
    ("0-24",   1,  24),
    ("24-72",  25, 72),
    ("72-120", 73, 120),
    ("120-168",121,168),
    ("168-240",169,240),
]

# Period stratification for eval / stratified hold-out.
PERIOD_BANDS = [
    ("<8s",    0.0,  8.0),
    ("8-11s",  8.0,  11.0),
    ("11-14s", 11.0, 14.0),
    (">14s",   14.0, 999.0),
]

# 8-way direction quadrants (wave-FROM convention, same as NDBC/Open-Meteo)
DIRECTION_QUADRANTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SEASONS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}


def ensure_dirs() -> None:
    """Create the standard .csc_data/* and .csc_models/ directories."""
    for d in (CSC_DATA_DIR, FORECASTS_DIR, OBSERVATIONS_DIR,
              PREDICTIONS_DIR, LIVE_LOG_DIR, AUDIT_DIR, CSC_MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
