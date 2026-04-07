"""
colesurfs — NOAA NDBC Buoy Fetcher

Spectral swell components are derived from two NDBC files:
  .data_spec  — non-directional spectral energy density (m²/Hz per frequency bin)
  .swdir      — mean wave direction (alpha1) per frequency bin

The algorithm reproduces Surfline's "Individual Swells" processing:
  1. Find local energy maxima in the swell band (period ≥ 6 s, freq ≤ 0.167 Hz).
     One extra bin beyond the cutoff is included so the edge bin has a right-
     neighbour for peak detection.
  2. Merge adjacent peaks that share similar direction (< 35°) AND have a high
     valley/min-peak ratio (≥ 0.70) — this prevents splitting a single broad
     swell train into spurious sub-peaks.
  3. Assign each swell-band frequency bin to the nearest (by energy valley) merged
     peak, forming partitions.
  4. For each partition compute:
       Hm0 = 4 √(Σ E(f) Δf)                  [significant wave height]
       Tm  = Σ(E(f)Δf · T(f)) / Σ(E(f)Δf)    [energy-weighted mean period]
       dir = circular energy-weighted mean of alpha1(f)
  5. Filter: Hm0 ≥ 0.2 ft and Tm ≥ 6 s; sort by energy (Hm0²·Tm); return top 2.

Validation against Surfline buoy 44097 (Block Island) at 0600 UTC 2026-03-25:
  Algorithm → Surfline
  1.56 ft  9.7 s  113°  →  1.60 ft  10 s  105°  ESE  ✓
  0.61 ft  6.4 s  104°  →  0.60 ft   6 s  100°    E  ✓

Falls back to .spec summary file if .data_spec/.swdir are unavailable (some buoys
only report the summary).
"""
import math
import requests
from datetime import datetime, timezone
from cache import ttl_cache
from config import m_to_ft, ms_to_mph, ms_to_kts

NDBC_URL          = "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
NDBC_LATEST_URL   = "https://www.ndbc.noaa.gov/data/latest_obs/{station_id}.txt"
NDBC_SPEC_URL     = "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
NDBC_DATA_SPEC_URL= "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.data_spec"
NDBC_SWDIR_URL    = "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir"
_FILL = {99.0, 999.0, 9999.0, 99.00, 999.00, 9999.00}

# Cardinal-to-degrees lookup for NDBC files that report direction as text
_CARD = {
    'N':0,'NNE':22,'NE':45,'ENE':67,'E':90,'ESE':112,'SE':135,'SSE':157,
    'S':180,'SSW':202,'SW':225,'WSW':247,'W':270,'WNW':292,'NW':315,'NNW':337,
}


def _safe(val: str):
    if val in ("MM", "m", None, ""):
        return None
    try:
        f = float(val)
        return None if f in _FILL else f
    except (ValueError, TypeError):
        return None


def _safe_dir(val: str):
    """Parse a direction value that may be numeric degrees or a cardinal string."""
    if val in ("MM", "m", None, ""):
        return None
    # Try numeric degrees first
    deg = _safe(val)
    if deg is not None:
        return deg
    # Fall back to cardinal string lookup (e.g. "ESE", "NNW")
    return _CARD.get(str(val).strip().upper())


def _parse(text: str) -> dict | None:
    lines   = [l.rstrip() for l in text.strip().split("\n")]
    headers = None
    data_lines = []
    for line in lines:
        if line.startswith("#"):
            if headers is None:
                headers = line.lstrip("# ").split()
        elif line.strip():
            data_lines.append(line)

    if not headers or not data_lines:
        return None

    # Find the most recent row that has a valid (non-MM) wave height.
    # The realtime2 file has ~45 days of hourly rows, newest first.
    # Surfline and other services show the latest valid reading, not strictly
    # the top row — so we do the same rather than surfacing a stale MM.
    row = None
    for line in data_lines:
        parts = line.split()
        candidate = dict(zip(headers, parts))
        if _safe(candidate.get("WVHT")) is not None:
            row = candidate
            break

    # Fall back to the top row if no row has wave height (all-MM file)
    if row is None:
        row = dict(zip(headers, data_lines[0].split()))

    try:
        yr = int(row.get("YY", row.get("#YY", "24")))
        if yr < 100:
            yr += 2000
        ts = datetime(
            yr, int(row.get("MM", 1)), int(row.get("DD", 1)),
            int(row.get("hh", 0)), int(row.get("mm", 0)),
            tzinfo=timezone.utc,
        )
    except Exception:
        ts = None

    wvht_m = _safe(row.get("WVHT"))
    dpd    = _safe(row.get("DPD"))
    # Do NOT fall back to APD — average period blends all wind chop into the
    # calculation and produces misleadingly short periods (e.g. 3s when the
    # dominant swell is 9s). Surfline also uses DPD only.
    mwd    = _safe_dir(row.get("MWD"))
    wspd   = _safe(row.get("WSPD"))
    wdir   = _safe(row.get("WDIR"))
    gst    = _safe(row.get("GST"))
    wtmp   = _safe(row.get("WTMP"))
    pres   = _safe(row.get("PRES"))

    wvht_ft = m_to_ft(wvht_m)
    period  = dpd  # dominant period only; None when DPD=MM
    energy  = round(wvht_ft ** 2 * period, 1) if (wvht_ft and period) else None

    return {
        "timestamp":          ts.isoformat() if ts else None,
        "wave_height_ft":     wvht_ft,
        "wave_period_s":      period,
        "wave_direction_deg": mwd,
        "energy":             energy,
        "wind_speed_kts":     ms_to_kts(wspd),
        "wind_direction_deg": wdir,
        "wind_gust_kts":      ms_to_kts(gst),
        "wind_speed_mph":     ms_to_mph(wspd),
        "wind_gust_mph":      ms_to_mph(gst),
        "water_temp_c":       wtmp,
        "pressure_hpa":       pres,
    }


def _parse_spectral_file(text: str, value_offset: int) -> list[tuple[float, float]]:
    """
    Generic parser for NDBC spectral files (.data_spec, .swdir).

    Both files have the same row structure:
      YY MM DD hh mm [sep_freq]  val1 (freq1)  val2 (freq2) ...

    value_offset = 1 → skip one extra column after the timestamp (sep_freq in .data_spec)
    value_offset = 0 → no extra column (.swdir, .swdir2, .swr1, .swr2)

    Returns list of (freq, value) pairs from the most recent valid data row,
    or [] if the file cannot be parsed.
    """
    lines = [l.rstrip() for l in text.strip().split("\n")]
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    if not data_lines:
        return []
    parts   = data_lines[0].split()
    offset  = 5 + value_offset          # skip YY MM DD hh mm [sep_freq]
    bins: list[tuple[float, float]] = []
    i = offset
    while i + 1 < len(parts):
        try:
            val  = float(parts[i])
            freq = float(parts[i + 1].strip("()"))
            bins.append((freq, val))
        except ValueError:
            pass
        i += 2
    return bins


def _spectral_components(spec_bins: list, swdir_bins: list) -> list:
    """
    Compute individual swell components from raw NDBC spectral data.

    spec_bins  : [(freq, energy_m2_per_hz), ...]  from .data_spec
    swdir_bins : [(freq, direction_deg),    ...]  from .swdir

    Returns a list of component dicts (same schema as used by models.py) sorted
    by energy descending.  Only swell-band components (Tm ≥ 6 s, Hm0 ≥ 0.2 ft)
    are returned; at most 2 are kept.
    """
    # Align the two arrays by frequency index (they should match exactly)
    n = min(len(spec_bins), len(swdir_bins))
    if n == 0:
        return []

    freqs  = [spec_bins[i][0]  for i in range(n)]
    energy = [spec_bins[i][1]  for i in range(n)]
    dirs   = [swdir_bins[i][1] for i in range(n)]

    # Centred-difference bin widths (m Hz⁻¹ → m² when multiplied by spectral density)
    def bw(i: int) -> float:
        if i == 0:   return freqs[1] - freqs[0]
        if i == n-1: return freqs[-1] - freqs[-2]
        return (freqs[i + 1] - freqs[i - 1]) / 2.0

    # Swell cutoff: period ≥ 6 s  ↔  freq ≤ 1/6 ≈ 0.1667 Hz
    # Include one extra bin beyond the cutoff so the last swell-band bin has a
    # right-neighbour for peak detection, then restrict partitions to the cutoff.
    SWELL_CUTOFF = 1.0 / 6.0                                # 0.1667 Hz
    ext_idx  = [i for i in range(n) if freqs[i] <= SWELL_CUTOFF + 0.015]
    swell_idx = [i for i in range(n) if freqs[i] <= SWELL_CUTOFF]

    # ── 1. Find local energy maxima ──────────────────────────────────────────
    NOISE_FLOOR = 0.005   # m²/Hz — ignore sub-noise peaks
    raw_peaks: list[int] = []
    for pos in range(1, len(ext_idx) - 1):
        i  = ext_idx[pos]
        pi = ext_idx[pos - 1]
        ni = ext_idx[pos + 1]
        if energy[i] > energy[pi] and energy[i] > energy[ni] and energy[i] > NOISE_FLOOR:
            raw_peaks.append(i)

    if not raw_peaks:
        return []

    # ── 2. Merge adjacent peaks that form one swell train ───────────────────
    # Criterion: merge if the peaks share similar direction (< DIR_THRESH°) AND
    # the valley between them is ≥ VALLEY_THRESH of the smaller peak's energy.
    # Physically: the direction test keeps separate swells from different storms
    # apart even if their spectra overlap; the valley test keeps the merge from
    # combining clearly distinct systems that happen to be co-directional.
    DIR_THRESH    = 35.0   # degrees
    VALLEY_THRESH = 0.70   # fraction

    def _dir_diff(a: float, b: float) -> float:
        d = abs(a - b) % 360
        return min(d, 360.0 - d)

    merged: list[int] = [raw_peaks[0]]
    for pk in raw_peaks[1:]:
        prev      = merged[-1]
        valley_e  = min(energy[j] for j in range(prev, pk + 1))
        min_peak  = min(energy[prev], energy[pk])
        ratio     = valley_e / min_peak if min_peak > 0 else 0.0
        d_diff    = _dir_diff(dirs[prev], dirs[pk])
        if d_diff < DIR_THRESH and ratio >= VALLEY_THRESH:
            # keep the higher-energy bin as the partition representative
            merged[-1] = pk if energy[pk] > energy[prev] else prev
        else:
            merged.append(pk)

    # ── 3. Assign swell-band bins to partitions via valley boundaries ────────
    def _partition_bins(peak_i: int) -> list[int]:
        rank = merged.index(peak_i)
        # left edge: one bin past the valley between previous peak and this one
        li = 0
        if rank > 0:
            prev_pk = merged[rank - 1]
            li = min(range(prev_pk, peak_i + 1), key=lambda j: energy[j]) + 1
        # right edge: the valley bin between this peak and the next
        ri = max(swell_idx) if swell_idx else 0
        if rank < len(merged) - 1:
            nxt_pk = merged[rank + 1]
            ri = min(range(peak_i, nxt_pk + 1), key=lambda j: energy[j])
        return [j for j in swell_idx if li <= j <= ri]

    # ── 4 & 5. Compute Hm0, Tm, direction for each partition ────────────────
    def _circular_mean(weights: list[float], angles_deg: list[float]) -> float:
        ss = sum(w * math.sin(math.radians(a)) for w, a in zip(weights, angles_deg))
        cs = sum(w * math.cos(math.radians(a)) for w, a in zip(weights, angles_deg))
        return math.degrees(math.atan2(ss, cs)) % 360.0

    MIN_HM0_FT = 0.2
    components: list[dict] = []

    for pk in merged:
        if pk not in swell_idx:
            continue   # peak itself is outside the swell band
        part = _partition_bins(pk)
        if not part:
            continue

        w       = [energy[i] * bw(i) for i in part]   # energy per bin (m²)
        total_e = sum(w)
        if total_e <= 0:
            continue

        hm0_m   = 4.0 * math.sqrt(total_e)
        hm0_ft  = m_to_ft(hm0_m)
        # Energy-weighted mean period — matches Surfline's displayed period
        Tm      = sum(wi * (1.0 / freqs[i]) for wi, i in zip(w, part)) / total_e
        mean_dir = _circular_mean(w, [dirs[i] for i in part])
        e_score  = round(hm0_ft ** 2 * Tm, 1)

        if hm0_ft < MIN_HM0_FT or Tm < 6.0:
            continue

        components.append({
            "height_ft":     round(hm0_ft, 2),
            "period_s":      round(Tm, 1),
            "direction_deg": round(mean_dir),
            "energy":        e_score,
            "type":          "swell",
        })

    components.sort(key=lambda c: c["energy"] or 0, reverse=True)
    return components[:2]


def _parse_spec(text: str) -> list:
    """
    Parse NDBC spectral wave summary (.spec) file into a list of swell components.

    The .spec file separates the sea state into:
      - Primary swell:  SwH (m), SwP (s), SwD (deg)
      - Wind sea:       WWH (m), WWP (s), WWD (deg)

    Returns a list (0–2 items) sorted by period descending, after filtering
    components below 6 s (consistent with the noise floor in models.py).
    """
    lines = [l.rstrip() for l in text.strip().split("\n")]
    headers = None
    data_lines = []
    for line in lines:
        if line.startswith("#"):
            if headers is None:
                headers = line.lstrip("# ").split()
        elif line.strip():
            data_lines.append(line)

    if not headers or not data_lines:
        return []

    # Scan for the most recent row that has at least one valid spectral value.
    # The .spec file is newest-first (same as realtime2.txt), and the top row
    # can have MM across the board — blindly taking data_lines[0] would return
    # zero components even when fresh data exists a few rows down.
    row = None
    for line in data_lines:
        parts = line.split()
        candidate = dict(zip(headers, parts))
        if _safe(candidate.get("SwH")) is not None:
            row = candidate
            break

    # Fall back to top row if every row is all-MM
    if row is None:
        row = dict(zip(headers, data_lines[0].split()))

    components = []

    def _add(h_key, p_key, d_key, comp_type):
        h_m = _safe(row.get(h_key))
        p   = _safe(row.get(p_key))
        d   = _safe_dir(row.get(d_key))   # may be degrees or cardinal string e.g. "ESE"
        if not h_m or h_m <= 0.0:
            return
        if not p or p < 6.0:          # < 6 s → FLAT noise, skip
            return
        h_ft   = m_to_ft(h_m)
        energy = round(h_ft ** 2 * p, 1) if (h_ft and p) else None
        components.append({
            "height_ft":     h_ft,
            "period_s":      round(p, 1),
            "direction_deg": d,
            "energy":        energy,
            "type":          comp_type,
        })

    _add("SwH", "SwP", "SwD", "swell")   # wind sea (WWH/WWP/WWD) intentionally excluded

    # Highest energy first (height²×period) — energy is the truer measure of
    # wave power; pure period sort can put a small distant groundswell above a
    # larger, choppier local swell that actually matters more for surfing.
    components.sort(key=lambda c: c["energy"] or 0, reverse=True)
    return components


def _parse_spectral_file_all_rows(text: str, value_offset: int) -> dict:
    """
    Parse ALL rows from an NDBC spectral file (.data_spec or .swdir).
    Returns {iso_timestamp: [(freq, value), ...], ...} keyed by UTC timestamp.
    """
    lines = [l.rstrip() for l in text.strip().split("\n")]
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    result = {}
    offset = 5 + value_offset  # skip YY MM DD hh mm [sep_freq]
    for line in data_lines:
        parts = line.split()
        if len(parts) < offset + 2:
            continue
        try:
            yr = int(parts[0])
            if yr < 100:
                yr += 2000
            ts = datetime(yr, int(parts[1]), int(parts[2]),
                          int(parts[3]), int(parts[4]),
                          tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue
        bins = []
        i = offset
        while i + 1 < len(parts):
            try:
                val  = float(parts[i])
                freq = float(parts[i + 1].strip("()"))
                bins.append((freq, val))
            except ValueError:
                pass
            i += 2
        if bins:
            result[ts.isoformat()] = bins
    return result


def _fetch_historical_spectral(station_id: str, cutoff_dt: datetime) -> dict:
    """
    Fetch .data_spec + .swdir for all available rows and compute spectral
    components for each timestamp after cutoff_dt.
    Returns {iso_timestamp: [component_dicts], ...}.
    """
    try:
        rds = requests.get(NDBC_DATA_SPEC_URL.format(station_id=station_id),
                           timeout=20, headers={"User-Agent": "ColeSurfs/1.0"})
        rsw = requests.get(NDBC_SWDIR_URL.format(station_id=station_id),
                           timeout=20, headers={"User-Agent": "ColeSurfs/1.0"})
        if rds.status_code != 200 or rsw.status_code != 200:
            return {}
    except Exception:
        return {}

    spec_all  = _parse_spectral_file_all_rows(rds.text, value_offset=1)
    swdir_all = _parse_spectral_file_all_rows(rsw.text, value_offset=0)
    cutoff_iso = cutoff_dt.isoformat()

    result = {}
    for ts_key in spec_all:
        if ts_key < cutoff_iso:
            continue
        if ts_key not in swdir_all:
            continue
        comps = _spectral_components(spec_all[ts_key], swdir_all[ts_key])
        if comps:
            result[ts_key] = comps
    return result


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_buoy_history(station_id: str, days: int = 5) -> dict | None:
    """
    Fetch the last `days` of hourly buoy observations from NDBC realtime2,
    plus spectral swell components for each timestamp where spectral data exists.

    Energy is computed as height_ft * period_s**2 (wave energy proxy specified
    for the historical chart — intentionally differs from the height**2 * period
    scoring used elsewhere in the app).
    """
    from datetime import timedelta
    url = NDBC_URL.format(station_id=station_id)
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"[buoy_history] {station_id}: fetch failed — {type(e).__name__}: {e}")
        return None

    lines = [l.rstrip() for l in r.text.strip().split("\n")]
    headers = None
    data_lines = []
    for line in lines:
        if line.startswith("#"):
            if headers is None:
                headers = line.lstrip("# ").split()
        elif line.strip():
            data_lines.append(line)

    if not headers or not data_lines:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    records = []
    for line in data_lines:
        parts = line.split()
        row = dict(zip(headers, parts))
        try:
            yr = int(row.get("YY", row.get("#YY", "24")))
            if yr < 100:
                yr += 2000
            ts = datetime(yr, int(row.get("MM", 1)), int(row.get("DD", 1)),
                          int(row.get("hh", 0)), int(row.get("mm", 0)),
                          tzinfo=timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            break  # file is newest-first

        wvht_m = _safe(row.get("WVHT"))
        dpd    = _safe(row.get("DPD"))
        mwd    = _safe_dir(row.get("MWD"))
        wvht_ft = m_to_ft(wvht_m)

        energy = round(wvht_ft * dpd ** 2, 1) if (wvht_ft and dpd) else None

        records.append({
            "timestamp":     ts.isoformat(),
            "wave_height_ft": wvht_ft,
            "wave_period_s":  dpd,
            "wave_direction_deg": mwd,
            "energy":        energy,
            "components":    [],  # filled below if spectral data available
        })

    records.reverse()  # oldest-first for charting

    # Merge spectral components
    try:
        spec_map = _fetch_historical_spectral(station_id, cutoff)
        for rec in records:
            ts_key = rec["timestamp"]
            if ts_key in spec_map:
                rec["components"] = spec_map[ts_key]
    except Exception as e:
        print(f"[buoy_history] {station_id}: spectral merge error — {type(e).__name__}: {e}")

    from cache import record_api_calls
    record_api_calls("NOAA_buoy_history", 1)

    return {"station_id": station_id, "records": records}


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_buoy(station_id: str) -> dict | None:
    """Try realtime2 first, fall back to latest_obs if that fails or parses empty."""
    for url_tmpl in [NDBC_URL, NDBC_LATEST_URL]:
        src = "realtime2" if "realtime2" in url_tmpl else "latest_obs"
        url = url_tmpl.format(station_id=station_id)
        try:
            r = requests.get(url, timeout=15,
                             headers={"User-Agent": "ColeSurfs/1.0"})
            r.raise_for_status()
        except Exception as e:
            print(f"[buoy] {station_id} ({src}): fetch failed — {type(e).__name__}: {e}")
            continue  # try next URL

        result = _parse(r.text)
        if result is None:
            print(f"[buoy] {station_id} ({src}): parse returned None — "
                  f"first 120 chars: {r.text[:120]!r}")
            continue  # try next URL

        wvht = result.get("wave_height_ft")
        wspd = result.get("wind_speed_kts")
        if wvht is None:
            # Entire file had no valid WVHT — try the other URL
            print(f"[buoy] {station_id} ({src}): all rows MM for wave height, trying next URL")
            continue
        print(f"[buoy] {station_id} ({src}): OK — {wvht}ft @ {result.get('wave_period_s')}s")

        # ── Fetch individual swell components ─────────────────────────────
        # Preferred: raw spectral files (.data_spec + .swdir) → Surfline-equivalent
        # Fallback:  spectral summary (.spec) → 1 swell only
        comps: list = []
        try:
            rds = requests.get(NDBC_DATA_SPEC_URL.format(station_id=station_id),
                               timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
            rsw = requests.get(NDBC_SWDIR_URL.format(station_id=station_id),
                               timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
            if rds.status_code == 200 and rsw.status_code == 200:
                spec_bins  = _parse_spectral_file(rds.text, value_offset=1)
                swdir_bins = _parse_spectral_file(rsw.text, value_offset=0)
                if spec_bins and swdir_bins:
                    comps = _spectral_components(spec_bins, swdir_bins)
                    print(f"[buoy] {station_id} spectral: {len(comps)} component(s) "
                          f"(data_spec+swdir)")
                else:
                    print(f"[buoy] {station_id} spectral files empty, trying .spec")
            else:
                print(f"[buoy] {station_id} spectral files HTTP "
                      f"{rds.status_code}/{rsw.status_code}, trying .spec")
        except Exception as e:
            print(f"[buoy] {station_id} spectral fetch error — {type(e).__name__}: {e}")

        if not comps:
            # Fall back to .spec summary (only gives 1 primary swell partition)
            try:
                rs = requests.get(NDBC_SPEC_URL.format(station_id=station_id),
                                  timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
                rs.raise_for_status()
                comps = _parse_spec(rs.text)
                print(f"[buoy] {station_id} .spec fallback: {len(comps)} component(s)")
            except Exception as e:
                print(f"[buoy] {station_id} .spec fallback failed — {type(e).__name__}: {e}")

        result["components"] = comps

        return result

    print(f"[buoy] {station_id}: all URLs exhausted, returning offline marker")
    return {"_offline": True, "buoy_id": station_id}
