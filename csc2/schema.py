"""CSC2 buoy scope, data layout, and schema constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSC2_DATA_DIR = PROJECT_ROOT / ".csc2_data"
CSC2_MODELS_DIR = PROJECT_ROOT / ".csc2_models"

FORECASTS_DIR = CSC2_DATA_DIR / "forecasts"      # one parquet per cycle per buoy per model
OBSERVATIONS_DIR = CSC2_DATA_DIR / "observations"
LOGS_DIR = CSC2_DATA_DIR / "logs"

# (buoy_id, label, lat, lon, scope)
BUOYS = [
    ("44013", "Boston (MA)",          42.346, -70.651, "east"),
    ("44065", "NY Harbor Entrance",   40.369, -73.703, "east"),
    ("44097", "Block Island Sound",   40.969, -71.127, "east"),
    ("44091", "Barnegat (NJ)",        39.778, -73.770, "east"),
    ("44098", "Jeffrey's Ledge (NH)", 42.798, -70.168, "east"),
    ("46025", "Santa Monica Basin",   33.755, -119.045, "west"),
    ("46221", "Santa Monica Bay",     33.855, -118.641, "west"),
    ("46268", "Topanga Nearshore",    34.032, -118.641, "west"),
]

BUOY_IDS = [b[0] for b in BUOYS]
EAST_BUOYS = [b[0] for b in BUOYS if b[4] == "east"]
WEST_BUOYS = [b[0] for b in BUOYS if b[4] == "west"]


def buoy_meta(buoy_id: str):
    for b in BUOYS:
        if b[0] == buoy_id:
            return {"buoy_id": b[0], "label": b[1],
                    "lat": b[2], "lon": b[3], "scope": b[4]}
    raise KeyError(f"unknown CSC2 buoy: {buoy_id!r}")


# Forecast record columns written by the live logger. Each row = one
# (buoy, model, cycle_utc, valid_utc) — i.e. one lead-time sample from one
# forecast cycle. This format preserves lead-time structure, which is
# non-negotiable for a forecast-correction model.
FORECAST_COLUMNS = [
    "buoy_id",                      # str  e.g. "44065"
    "model",                        # str  "EURO" | "GFS"
    "cycle_utc",                    # str ISO-8601 UTC of the forecast run
    "valid_utc",                    # str ISO-8601 UTC of the prediction target
    "lead_hours",                   # int  valid_utc - cycle_utc in hours
    # Dashboard-comparable swell components after processing pipeline
    "sw1_height_ft",                # primary-swell Hs (ft)
    "sw1_period_s",                 # primary-swell Tp-equivalent (s) — Tm01*1.20 for CMEMS
    "sw1_direction_deg",            # primary-swell mean-from direction (°)
    "sw2_height_ft",                # secondary-swell Hs (ft)  (may be NaN)
    "sw2_period_s",
    "sw2_direction_deg",
    # Combined-sea (informational only; not used as a "swell" per dashboard rules)
    "combined_height_m",
    "combined_period_s",
    "combined_direction_deg",
    "ingest_utc",                   # str ISO-8601 wall clock of this row
]


def ensure_dirs() -> None:
    for d in (CSC2_DATA_DIR, FORECASTS_DIR, OBSERVATIONS_DIR, LOGS_DIR,
              CSC2_MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
