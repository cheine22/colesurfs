"""
colesurfs — NOAA CO-OPS Tide Predictions
Fetches hourly + high/low tide predictions; applies per-spot Surfline-matched corrections.

Station IDs and per-spot time offsets are defined in regions.yaml (tide_station,
tide_hi_offset, tide_lo_offset fields on each spot entry).

Per-spot time corrections:
  Reverse-engineered from Surfline API hilo data on 2026-04-01.
  Applied to ALL tide data:
    • Hourly heights: interpolated from the NOAA series shifted by avg(hi, lo) offset,
      so height_ft and pct reflect the beach's actual tidal state at each hour.
    • hilo_time label: stamped using the per-type offset (hi or lo separately)
      for maximum accuracy on the exact high/low time displayed in the cell.
  Positive = Surfline is later than NOAA; negative = Surfline is earlier.
"""
import bisect
import requests
from datetime import datetime, timedelta
from cache import ttl_cache
from config import WIND_SPOTS, FORECAST_DAYS

NOAA_TIDES_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


@ttl_cache(ttl_seconds=3600)
def fetch_tide_predictions(past_days: int = 0) -> dict:
    """
    Fetch hourly + hilo tide predictions for all unique tide_station IDs in WIND_SPOTS.
    Returns per-spot annotated data with Surfline-matched time corrections applied.

    `past_days` (0..30) extends the begin_date backwards so the dashboard's
    historical-data toggle can populate tide cells in the -240 h strip. Since
    tides are harmonic predictions (not observations), NOAA serves them for
    any date window — past or future — with the same reliability.

    Returns:
      { spot_name: { "YYYY-MM-DDTHH:MM": {height_ft, pct, hilo_time?, hilo_type?} } }

    pct       = (height - daily_low) / (daily_high - daily_low) × 100  (from shifted heights)
    hilo_time = formatted string like "5:49am" — corrected by tide_hi/lo_offset
    hilo_type = "H" or "L"
    Times are in America/New_York local time (LST/LDT), matching Open-Meteo format.
    """
    past_days = max(0, min(int(past_days or 0), 30))
    today    = datetime.now()
    begin_dt = (today - timedelta(days=past_days)).strftime("%Y%m%d")
    end_dt   = (today + timedelta(days=FORECAST_DAYS - 1)).strftime("%Y%m%d")

    # Fetch raw data once per unique station
    station_ids = list({ws["tide_station"] for ws in WIND_SPOTS if ws.get("tide_station")})
    hourly_by_station: dict[str, list] = {}
    hilo_by_station:   dict[str, list] = {}
    for sid in station_ids:
        h, hl = _fetch_station(sid, begin_dt, end_dt)
        hourly_by_station[sid] = h
        hilo_by_station[sid]   = hl

    # Annotate per spot with that spot's own time offsets
    result: dict[str, dict] = {}
    for ws in WIND_SPOTS:
        sid = ws.get("tide_station")
        if not sid:
            continue
        hourly = hourly_by_station.get(sid, [])
        hilo   = hilo_by_station.get(sid, [])
        hi_off = ws.get("tide_hi_offset", 0)
        lo_off = ws.get("tide_lo_offset", 0)
        result[ws["name"]] = _annotate(hourly, hilo, hi_off, lo_off) if hourly else {}

    return result


# ── NOAA fetch helpers ────────────────────────────────────────────────────────

def _get(station_id: str, begin_date: str, end_date: str, interval: str) -> list:
    """Single NOAA CO-OPS request for predictions."""
    url = (
        f"{NOAA_TIDES_URL}"
        f"?station={station_id}"
        f"&product=predictions"
        f"&datum=MLLW"
        f"&time_zone=lst_ldt"
        f"&interval={interval}"
        f"&units=english"
        f"&begin_date={begin_date}"
        f"&end_date={end_date}"
        f"&format=json"
    )
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
        data  = r.json()
        preds = data.get("predictions", [])
        if not preds:
            print(f"[tide] no {interval} data for station {station_id}: "
                  f"{data.get('error', {})}")
        return preds
    except Exception as e:
        print(f"[tide] {interval} fetch failed for station {station_id}: {e}")
        return []


def _fetch_station(station_id: str, begin_date: str,
                   end_date: str) -> tuple[list, list]:
    """Return (hourly_preds, hilo_preds) for a station."""
    return (_get(station_id, begin_date, end_date, "h"),
            _get(station_id, begin_date, end_date, "hilo"))


# ── Interpolation helper ──────────────────────────────────────────────────────

def _build_interp(hourly: list) -> tuple[list, list]:
    """
    Parse NOAA hourly records into parallel sorted lists of (timestamps_sec, heights)
    for fast linear interpolation.
    """
    epoch = datetime(1970, 1, 1)
    ts, hs = [], []
    for p in hourly:
        dt = datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
        ts.append((dt - epoch).total_seconds())
        hs.append(float(p["v"]))
    return ts, hs


def _interp_height(ts: list, hs: list, target_dt: datetime) -> float:
    """
    Linearly interpolate (or clamp) the NOAA height series at target_dt.
    ts / hs are parallel sorted lists from _build_interp.
    """
    epoch  = datetime(1970, 1, 1)
    target = (target_dt - epoch).total_seconds()

    if target <= ts[0]:
        return hs[0]
    if target >= ts[-1]:
        return hs[-1]

    i = bisect.bisect_right(ts, target) - 1   # index of left bracket
    t0, t1 = ts[i], ts[i + 1]
    h0, h1 = hs[i], hs[i + 1]
    frac = (target - t0) / (t1 - t0)
    return h0 + frac * (h1 - h0)


# ── Annotation ────────────────────────────────────────────────────────────────

def _fmt_time(dt: datetime) -> str:
    """Format datetime as '5:49am' or '11:13am'."""
    h12  = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{h12}:{dt.minute:02d}{ampm}"


def _annotate(hourly: list, hilo: list,
              hi_offset_min: int = 0, lo_offset_min: int = 0) -> dict:
    """
    Build per-hourly-slot dict: {height_ft, pct, hilo_time?, hilo_type?}

    hi_offset_min / lo_offset_min:
      Minutes to shift NOAA predictions to match Surfline (beach) times.
      Derived by reverse-engineering Surfline predictions vs NOAA reference data.
      Negative = beach tide arrives earlier than the gauge predicts.

    Height correction (all hourly slots):
      For each output hour T, sample the NOAA series at T − avg_offset.
      This gives the height the beach actually experiences at T rather than
      the height the reference gauge reads at T.
      avg_offset = (hi_offset + lo_offset) / 2  — a sound approximation since
      the tidal phase varies smoothly between H and L extremes.

    Hilo annotation:
      The exact H/L type-specific offsets are used for the hilo_time label
      so the displayed time is as accurate as possible.

    Algorithm:
      1. Build interpolator from raw NOAA hourly series.
      2. For each output slot T, evaluate height at T − avg_offset_min.
      3. Compute daily high/low from the corrected heights for pct.
      4. For each hilo event, apply the per-type offset, find the nearest
         corrected hourly slot, and stamp hilo_time + hilo_type.
    """
    avg_offset = (hi_offset_min + lo_offset_min) / 2.0

    # ── Step 1: build interpolator ────────────────────────────────────────────
    ts_arr, hs_arr = _build_interp(hourly)

    # ── Step 2: corrected height per output slot ──────────────────────────────
    # Output slots are the same calendar times as the NOAA hourly records;
    # we just sample the series at (slot_time − avg_offset).
    slot_times: list[tuple[str, datetime]] = []
    for p in hourly:
        slot_dt = datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
        noaa_lookup_dt = slot_dt - timedelta(minutes=avg_offset)
        h = _interp_height(ts_arr, hs_arr, noaa_lookup_dt)
        slot_times.append((p["t"], slot_dt, h))

    # ── Step 3: daily high/low from corrected heights → pct ──────────────────
    by_date: dict[str, list] = {}
    for t_str, slot_dt, h in slot_times:
        by_date.setdefault(t_str[:10], []).append((t_str, h))

    result: dict[str, dict] = {}
    slot_dts: dict[str, datetime] = {}
    for recs in by_date.values():
        heights    = [h for _, h in recs]
        day_low    = min(heights)
        day_high   = max(heights)
        tide_range = day_high - day_low
        for t_str, h in recs:
            pct = round((h - day_low) / tide_range * 100) if tide_range > 0.01 else 50
            key = t_str.replace(" ", "T")
            result[key] = {"height_ft": round(h, 1), "pct": pct}
            slot_dts[key] = datetime.strptime(key, "%Y-%m-%dT%H:%M")

    # ── Step 4: stamp corrected hilo times onto nearest hourly slot ───────────
    for event in hilo:
        try:
            ev_dt   = datetime.strptime(event["t"], "%Y-%m-%d %H:%M")
            ev_type = event.get("type", "")   # "H" or "L"
        except (ValueError, KeyError):
            continue

        offset  = hi_offset_min if ev_type == "H" else lo_offset_min
        ev_corr = ev_dt + timedelta(minutes=offset)

        best_key  = None
        best_diff = timedelta(minutes=31)
        for key, slot_dt in slot_dts.items():
            diff = abs(ev_corr - slot_dt)
            if diff < best_diff:
                best_diff = diff
                best_key  = key

        if best_key:
            result[best_key]["hilo_time"] = _fmt_time(ev_corr)
            result[best_key]["hilo_type"] = ev_type

    return result
