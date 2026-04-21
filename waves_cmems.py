"""
colesurfs — Copernicus Marine Service (CMEMS) EURO wave forecast fetcher.

Pulls the ECMWF WAM analysis+forecast product
  cmems_mod_glo_wav_anfc_0.083deg_PT3H-i
which publishes swell-partition fields (VHM0_SW1/SW2, VTM01_SW1/SW2,
VMDR_SW1/SW2) that Open-Meteo's ecmwf_wam025 endpoint does not expose.

Per-buoy geographic granularity: one CMEMS point-subset per region buoy.
All spots in a region share their region's buoy-location forecast. CMEMS
native cadence is 3-hourly; we linearly interpolate magnitudes and
circularly interpolate from-directions to produce an hourly grid that
matches the dashboard's display resolution.

Output record shape is identical to waves._parse_response so the frontend
treats CMEMS as a drop-in replacement.

v1.5: introduced alongside Open-Meteo EURO as the "C-EURO" model.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone as dtz
from zoneinfo import ZoneInfo

from cache import ttl_cache, record_api_calls
from config import FORECAST_DAYS, TIMEZONE, SPOTS, m_to_ft

# copernicusmarine logs every subset at INFO plus a WARNING about "subset
# exceeds coords" when we pad the window past the forecast tail. Silence
# those — the [cmems] print lines below are the signal we care about.
logging.getLogger("copernicusmarine").setLevel(logging.ERROR)
logging.getLogger("copernicus_marine_client").setLevel(logging.ERROR)


CMEMS_PRODUCT = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
CMEMS_VARS = [
    "VHM0", "VTPK", "VMDR",
    "VHM0_SW1", "VTM01_SW1", "VMDR_SW1",
    "VHM0_SW2", "VTM01_SW2", "VMDR_SW2",
]

# Long TTL so an occasional CMEMS outage does not wipe the cache before
# the 30-minute warmer can refresh. skip_none=True keeps the last-known-good
# payload in place when a fetch fails.
_CMEMS_TTL_SECONDS = 86400

# CMEMS publishes VTM01_SW1/SW2 (spectral mean period m1) but no VTPK_SW*.
# Windy/Surfline/buoy-DPD use peak period Tp. For typical ocean swell
# spectra Tm01/Tp ≈ 0.83, so we scale Tm01 by 1/0.83 ≈ 1.20 for display.
_TM01_TO_TP = 1.20

_NY = ZoneInfo(TIMEZONE)


def _safe(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _forecast_window_utc() -> tuple[datetime, datetime]:
    """Today 00:00 New-York → +FORECAST_DAYS, expressed in UTC."""
    now_local = datetime.now(_NY)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=FORECAST_DAYS)
    return start_local.astimezone(dtz.utc), end_local.astimezone(dtz.utc)


def _lerp(a, b, f):
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    return a * (1 - f) + b * f


def _circ_lerp(a, b, f):
    """Interpolate two from-directions along the shorter arc."""
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    ra = math.radians(a); rb = math.radians(b)
    x = (1 - f) * math.cos(ra) + f * math.cos(rb)
    y = (1 - f) * math.sin(ra) + f * math.sin(rb)
    return math.degrees(math.atan2(y, x)) % 360.0


def _interpolate_to_hourly(raw: list[dict]) -> list[dict]:
    """Expand a sorted list of dicts at 3-hour spacing into 1-hour rows.
    Each dict carries a 'utc' datetime and CMEMS_VARS keys."""
    if not raw:
        return []
    out = [raw[0]]
    for i in range(len(raw) - 1):
        a = raw[i]
        b = raw[i + 1]
        gap = int(round((b["utc"] - a["utc"]).total_seconds() / 3600))
        if gap <= 1:
            out.append(b)
            continue
        for k in range(1, gap):
            frac = k / gap
            row = {"utc": a["utc"] + timedelta(hours=k)}
            for v in CMEMS_VARS:
                if v.startswith("VMDR"):
                    row[v] = _circ_lerp(a.get(v), b.get(v), frac)
                else:
                    row[v] = _lerp(a.get(v), b.get(v), frac)
            out.append(row)
        out.append(b)
    return out


def _build_components(sw_h, sw_p, sw_d,
                       sw_h2, sw_p2, sw_d2):
    """Build up to 2 swell components from the SW1/SW2 partition fields.

    CMEMS publishes partition period as VTM01_* (spectral mean period m1),
    while Windy / Surfline / buoy-DPD display peak period Tp. For typical
    ocean spectra Tp ≈ Tm01 × (1 / 0.83). We apply a fixed 1.20 scalar to
    the partition period before display so the output aligns with those
    references. The underlying partition Hs and direction are unchanged.

    The combined-sea fallback that used to populate this list when all
    partitions were filtered was removed in v1.5 — we display swell only.
    """
    raw = [
        {"h_m": sw_h,  "p": sw_p,  "d": sw_d,  "type": "swell"},
        {"h_m": sw_h2, "p": sw_p2, "d": sw_d2, "type": "swell2"},
    ]
    comps = []
    for c in raw:
        h_m = _safe(c["h_m"])
        p_tm01 = _safe(c["p"])
        d = _safe(c["d"])
        if not h_m or h_m <= 0.0:
            continue
        # Filter wind chop at Tm01 < 5.0 s → Tp < ~6.0 s.
        if not p_tm01 or p_tm01 < 5.0:
            continue
        p_tp = p_tm01 * _TM01_TO_TP
        h_ft = m_to_ft(h_m)
        energy = round(h_ft ** 2 * p_tp, 1) if h_ft and p_tp else None
        comps.append({
            "height_ft": h_ft,
            "period_s": round(p_tp, 1),
            "direction_deg": d,
            "energy": energy,
            "type": c["type"],
        })

    comps.sort(key=lambda c: c["energy"] or 0, reverse=True)
    return comps[:2]


def _rows_to_records(rows: list[dict]) -> list[dict]:
    records = []
    for r in rows:
        t_local = r["utc"].astimezone(_NY)
        time_str = t_local.strftime("%Y-%m-%dT%H:%M")

        comps = _build_components(
            r.get("VHM0_SW1"), r.get("VTM01_SW1"), r.get("VMDR_SW1"),
            r.get("VHM0_SW2"), r.get("VTM01_SW2"), r.get("VMDR_SW2"),
        )
        primary = comps[0] if comps else None

        raw_dir = primary["direction_deg"] if primary else None
        if raw_dir is None:
            for key in ("VMDR_SW1", "VMDR_SW2", "VMDR"):
                v = _safe(r.get(key))
                if v is not None:
                    raw_dir = v
                    break

        records.append({
            "time": time_str,
            "wave_height_ft":     primary["height_ft"] if primary else None,
            "wave_period_s":      primary["period_s"] if primary else None,
            "wave_direction_deg": primary["direction_deg"] if primary else None,
            "energy":             primary["energy"] if primary else None,
            "components":         comps,
            "raw_direction_deg":  raw_dir,
            "combined_wave_height_m":      _safe(r.get("VHM0")),
            "combined_wave_period_s":      _safe(r.get("VTPK")),
            "combined_wave_direction_deg": _safe(r.get("VMDR")),
        })
    return records


def _open_dataset(lat: float, lon: float):
    """Ask copernicusmarine for a point subset of the ANFC forecast window.
    Returns an xarray.Dataset or raises."""
    try:
        import copernicusmarine as cm
    except ImportError as e:
        raise RuntimeError(
            "copernicusmarine not installed. `pip install copernicusmarine`"
        ) from e

    start_utc, end_utc = _forecast_window_utc()
    # Pad by 3h (one native CMEMS tick) on each side so hourly interpolation
    # has valid neighbors around the local-midnight window boundaries.
    query_start = start_utc - timedelta(hours=3)
    query_end = end_utc + timedelta(hours=3)
    pad = 0.12  # ~1.5 cells at 0.083°; enough to pick up a neighboring ocean cell
    return cm.open_dataset(
        dataset_id=CMEMS_PRODUCT,
        variables=CMEMS_VARS,
        minimum_longitude=lon - pad, maximum_longitude=lon + pad,
        minimum_latitude=lat - pad, maximum_latitude=lat + pad,
        start_datetime=query_start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=query_end.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def _extract_point_rows(ds, lat: float, lon: float) -> list[dict]:
    """From a 2D lat/lon × time xarray.Dataset, produce a list of raw rows
    (one per CMEMS 3-hourly step) at the nearest ocean cell to (lat, lon)."""
    import pandas as pd  # lazy: only when a fetch succeeds
    try:
        ds_pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
    except Exception:
        ds_pt = ds.squeeze()

    time_name = None
    for cand in ("time", "valid_time"):
        if cand in ds_pt.coords:
            time_name = cand
            break
    if time_name is None:
        raise RuntimeError(f"no time coord in CMEMS point; coords={list(ds_pt.coords)}")

    times = pd.to_datetime(ds_pt[time_name].values, utc=True)
    if len(times) == 0:
        return []

    rows: list[dict] = []
    for i, t in enumerate(times):
        row = {"utc": t.to_pydatetime()}
        for v in CMEMS_VARS:
            if v in getattr(ds_pt, "data_vars", {}):
                try:
                    val = ds_pt[v].values.reshape(-1)[i]
                    row[v] = None if pd.isna(val) else float(val)
                except Exception:
                    row[v] = None
            else:
                row[v] = None
        rows.append(row)
    rows.sort(key=lambda r: r["utc"])
    return rows


@ttl_cache(ttl_seconds=_CMEMS_TTL_SECONDS, skip_none=True)
def fetch_cmems_point(lat: float, lon: float) -> list | None:
    """Pull CMEMS wave forecast at a single lat/lon; return per-hour records
    in the waves._parse_response shape, or None on failure."""
    t0 = time.monotonic()
    record_api_calls("cmems_wave_forecast", 1)
    try:
        ds = _open_dataset(lat, lon)
    except Exception as e:
        print(f"[cmems] open_dataset ({lat:.3f},{lon:.3f}): {e}")
        return None

    try:
        raw = _extract_point_rows(ds, lat, lon)
    except Exception as e:
        print(f"[cmems] extract ({lat:.3f},{lon:.3f}): {e}")
        return None

    if not raw:
        return None

    hourly = _interpolate_to_hourly(raw)
    start_utc, end_utc = _forecast_window_utc()
    hourly = [r for r in hourly if start_utc <= r["utc"] < end_utc]
    records = _rows_to_records(hourly)
    elapsed = time.monotonic() - t0
    print(f"[cmems] ({lat:.3f},{lon:.3f}) {len(records)} rows in {elapsed:.1f}s")
    return records


def fetch_all_cmems_wave_forecasts() -> dict | None:
    """Parallel-fetch CMEMS for every region buoy in SPOTS. Returns
    {region_name: [hourly records]} or None if every buoy fails.

    Per-buoy results are cached individually by fetch_cmems_point, so this
    aggregator is cheap when all entries are warm."""
    t0 = time.monotonic()
    result: dict = {}
    n = len(SPOTS) or 1
    with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
        fut_to_spot = {
            pool.submit(fetch_cmems_point, s["lat"], s["lon"]): s for s in SPOTS
        }
        for fut in as_completed(fut_to_spot):
            s = fut_to_spot[fut]
            try:
                result[s["name"]] = fut.result(timeout=90)
            except Exception as e:
                print(f"[cmems_batch] {s['name']}: {e}")
                result[s["name"]] = None
    elapsed = time.monotonic() - t0
    any_ok = any(v is not None for v in result.values())
    ok_n = sum(1 for v in result.values() if v is not None)
    print(f"[cmems_batch] {ok_n}/{n} buoys OK in {elapsed:.1f}s")
    return result if any_ok else None
