"""
colesurfs — Sunrise/Sunset Calculator (local, no API dependency)

Uses the `astral` library to compute sunrise/sunset for the forecast period.
Replaces the browser-side Open-Meteo API call that was subject to 429 rate limiting.

Accuracy: within 1-2 minutes of NOAA solar calculator — more than sufficient
since the frontend pads sunrise/sunset by ±30 minutes for "first light"/"last light".
"""
from datetime import date, timedelta
from astral import LocationInfo
from astral.sun import sun
from zoneinfo import ZoneInfo

from config import FORECAST_DAYS


def compute_sun_data(lat: float, lon: float,
                     tz_name: str = "America/New_York",
                     days: int | None = None) -> dict:
    """
    Compute sunrise/sunset for a location over the forecast period.

    Returns:
      {
        "2026-04-07": {"sunrise": "2026-04-07T06:18", "sunset": "2026-04-07T19:32"},
        ...
      }

    Format matches what the frontend's fetchSunData() expects after parsing.
    """
    if days is None:
        days = FORECAST_DAYS

    tz = ZoneInfo(tz_name)
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name)

    result = {}
    today = date.today()
    for i in range(days):
        d = today + timedelta(days=i)
        try:
            s = sun(loc.observer, date=d, tzinfo=tz)
            result[d.isoformat()] = {
                "sunrise": s["sunrise"].strftime("%Y-%m-%dT%H:%M"),
                "sunset":  s["sunset"].strftime("%Y-%m-%dT%H:%M"),
            }
        except Exception:
            # Edge case: locations near poles can fail during polar day/night.
            # Not relevant for East Coast US but handle gracefully.
            pass

    return result
