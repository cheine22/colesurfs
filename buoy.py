"""
colesurfs — NOAA NDBC Buoy Fetcher

Spectral swell components are derived from two NDBC files:
  .data_spec  — non-directional spectral energy density (m²/Hz per frequency bin)
  .swdir      — mean wave direction (alpha1) per frequency bin

The algorithm reproduces Surfline's "Individual Swells" processing:
  1. Find local energy maxima over the WHOLE spectrum (no period cutoff —
     partitioning first and filtering afterwards is what keeps a 5–6 s
     windswell's energy in one piece instead of truncating it at a band edge).
  2. Merge adjacent peaks that share similar direction (< 35°) AND have a high
     valley/min-peak ratio (≥ 0.70) AND sit within a 1.5× frequency ratio —
     this prevents splitting a single broad swell train into spurious
     sub-peaks without letting a small long-period swell disappear into a
     big co-directional windsea on its shoulder.
  3. Assign each frequency bin to the nearest (by energy valley) merged peak,
     forming partitions.
  4. For each partition compute:
       Hm0 = 4 √(Σ E(f) Δf)      [significant wave height]
       Tp  = 1 / f_peak          [period of the partition's peak bin]
       dir = alpha1(f_peak)      [direction of the peak bin]
     Peak period and peak-bin direction are what Surfline prints; the
     energy-weighted mean period / circular-mean direction used until
     v1.13.3 read ~0.3 s long and 10–15° off on broad partitions.
  5. Filter: Hm0 ≥ 0.2 ft and Tp ≥ wave_common.MIN_SWELL_PERIOD_S (5.0 s, the
     same floor the wave models apply); sort by energy Hm0²·Tp (wave power
     ∝ H²T, the same proxy the wave models and Surfline use); return top 2.

Validation against Surfline, 2026-09-13 (algorithm → Surfline):
  44065 NY Harbor Entrance 1720Z   2.1 ft 5.6 s 124° → 2.2 ft 6 s 125°   0.9 ft 8.3 s 148° → 1.0 ft 8 s 150°
  44065 NY Harbor Entrance 1750Z   2.4 ft 5.3 s 136° → 2.3 ft 5 s 135°   0.7 ft 10.0 s 136° → 0.5 ft 10 s 140°
  44025 Long Island        1740Z   2.9 ft 5.9 s 144° → 2.8 ft 6 s 145°   0.9 ft 10.0 s 128° → 0.9 ft 10 s 130°
  (44097 Block Island agrees on period/direction but not height — Surfline's
  partition heights there don't sum to its own Hs, so it likely reads CDIP's
  2-D partitions for that buoy rather than NDBC's 1-D spectrum.)
(The pre-v1.13.3 code read 0.9 ft 9.3 s / 0.8 ft 6.7 s for the same spectrum:
it dropped every bin shorter than 6 s BEFORE partitioning.)

Falls back to .spec summary file if .data_spec/.swdir are unavailable (some buoys
only report the summary).
"""
import bisect
import math
import requests
from datetime import datetime, timedelta, timezone
import swell_rules
from cache import ttl_cache, record_api_calls
from config import m_to_ft, ms_to_mph, ms_to_kts
from wave_common import MIN_SWELL_PERIOD_S

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


def _split_ndbc(text: str) -> tuple[list | None, list[str]]:
    """(headers, data_lines) for an NDBC text file — the first '#' line is
    the header, later '#' lines (units) are skipped."""
    headers = None
    data_lines = []
    for line in text.strip().split("\n"):
        line = line.rstrip()
        if line.startswith("#"):
            if headers is None:
                headers = line.lstrip("# ").split()
        elif line.strip():
            data_lines.append(line)
    return headers, data_lines


def _row_ts(row: dict) -> datetime:
    """UTC timestamp from a stdmet row's YY MM DD hh mm columns."""
    yr = int(row.get("YY", row.get("#YY", "24")))
    if yr < 100:
        yr += 2000
    return datetime(
        yr, int(row.get("MM", 1)), int(row.get("DD", 1)),
        int(row.get("hh", 0)), int(row.get("mm", 0)),
        tzinfo=timezone.utc,
    )


def _parse_bins(parts: list[str], offset: int) -> list[tuple[float, float]]:
    """`val (freq)` pairs from a spectral-file row, from column `offset` on."""
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


def _parse(text: str) -> dict | None:
    headers, data_lines = _split_ndbc(text)
    if not headers or not data_lines:
        return None

    # Find the most recent row that has valid WVHT *and* DPD.  Requiring both
    # keeps "Buoy Now" in sync with the history chart, which filters on
    # energy != null (energy = wvht_ft² × dpd) — same criterion.
    # Falls back to a WVHT-only row if no row has DPD, and to the top row
    # if the file has no wave data at all.
    row = None
    wvht_fallback = None
    for line in data_lines:
        parts = line.split()
        candidate = dict(zip(headers, parts))
        if _safe(candidate.get("WVHT")) is not None:
            if wvht_fallback is None:
                wvht_fallback = candidate   # first row with valid WVHT
            if _safe(candidate.get("DPD")) is not None:
                row = candidate             # prefer row with both WVHT and DPD
                break

    if row is None:
        row = wvht_fallback
    # Fall back to the top row if no row has wave height (all-MM file)
    if row is None:
        row = dict(zip(headers, data_lines[0].split()))

    try:
        ts = _row_ts(row)
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
    # Wave energy proxy = height² × period (deep-water wave power ∝ H²T; the
    # convention Surfline's energy figure follows). Same formula in
    # fetch_buoy_history, _spectral_components and wave_common (models).
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
    return _parse_bins(data_lines[0].split(), 5 + value_offset)   # skip YY MM DD hh mm [sep_freq]


def _spectral_components(spec_bins: list, swdir_bins: list) -> list:
    """
    Compute individual swell components from raw NDBC spectral data.

    spec_bins  : [(freq, energy_m2_per_hz), ...]  from .data_spec
    swdir_bins : [(freq, direction_deg),    ...]  from .swdir

    Returns a list of component dicts (same schema as the wave-model
    `components`, see wave_common.build_swell_components) sorted by energy
    descending. The whole spectrum is partitioned first; only partitions with
    Tp ≥ MIN_SWELL_PERIOD_S and Hm0 ≥ 0.2 ft are returned, at most 2.
    """
    # Align the two arrays by frequency index (they should match exactly)
    n = min(len(spec_bins), len(swdir_bins))
    if n < 3:
        return []

    freqs  = [spec_bins[i][0]  for i in range(n)]
    energy = [spec_bins[i][1]  for i in range(n)]
    dirs   = [None if swdir_bins[i][1] in _FILL else swdir_bins[i][1] for i in range(n)]

    # Centred-difference bin widths (m Hz⁻¹ → m² when multiplied by spectral density)
    def bw(i: int) -> float:
        if i == 0:   return freqs[1] - freqs[0]
        if i == n-1: return freqs[-1] - freqs[-2]
        return (freqs[i + 1] - freqs[i - 1]) / 2.0

    # ── 1. Find local energy maxima over the whole spectrum ─────────────────
    NOISE_FLOOR = 0.005   # m²/Hz — ignore sub-noise peaks
    raw_peaks: list[int] = [
        i for i in range(1, n - 1)
        if energy[i] > energy[i - 1] and energy[i] > energy[i + 1] and energy[i] > NOISE_FLOOR
    ]
    if not raw_peaks:
        return []

    # ── 2. Merge adjacent peaks that form one swell train ───────────────────
    # Criterion: merge if the peaks share similar direction (< DIR_THRESH°) AND
    # the valley between them is ≥ VALLEY_THRESH of the smaller peak's energy.
    # Physically: the direction test keeps separate swells from different storms
    # apart even if their spectra overlap; the valley test keeps the merge from
    # combining clearly distinct systems that happen to be co-directional.
    # A peak with no direction (NDBC 999 fill) is merged on the valley test alone.
    # Peaks further apart than MAX_FREQ_RATIO in frequency are never one train:
    # the valley test is relative to the SMALLER peak, so a 0.5 ft 10 s swell
    # riding the shoulder of a 2.5 ft 5 s windsea (0.100 vs 0.190 Hz, both SE)
    # would otherwise vanish into it — Surfline reports the two separately.
    DIR_THRESH     = 35.0   # degrees
    VALLEY_THRESH  = 0.70   # fraction
    MAX_FREQ_RATIO = 1.5

    def _dir_diff(a: float | None, b: float | None) -> float:
        if a is None or b is None:
            return 0.0
        d = abs(a - b) % 360
        return min(d, 360.0 - d)

    merged: list[int] = [raw_peaks[0]]
    for pk in raw_peaks[1:]:
        prev      = merged[-1]
        valley_e  = min(energy[j] for j in range(prev, pk + 1))
        min_peak  = min(energy[prev], energy[pk])
        ratio     = valley_e / min_peak if min_peak > 0 else 0.0
        near      = freqs[pk] / freqs[prev] <= MAX_FREQ_RATIO
        if near and _dir_diff(dirs[prev], dirs[pk]) < DIR_THRESH and ratio >= VALLEY_THRESH:
            # keep the higher-energy bin as the partition representative
            merged[-1] = pk if energy[pk] > energy[prev] else prev
        else:
            merged.append(pk)

    # ── 3. Assign bins to partitions via valley boundaries ──────────────────
    def _partition_bins(rank: int) -> list[int]:
        peak_i = merged[rank]
        li = 0
        if rank > 0:
            prev_pk = merged[rank - 1]
            li = min(range(prev_pk, peak_i + 1), key=lambda j: energy[j]) + 1
        ri = n - 1
        if rank < len(merged) - 1:
            nxt_pk = merged[rank + 1]
            ri = min(range(peak_i, nxt_pk + 1), key=lambda j: energy[j])
        return list(range(li, ri + 1))

    # ── 4 & 5. Hm0, Tp, direction for each partition; filter; rank ──────────
    def _circular_mean(weights: list[float], angles_deg: list[float]) -> float:
        ss = sum(w * math.sin(math.radians(a)) for w, a in zip(weights, angles_deg))
        cs = sum(w * math.cos(math.radians(a)) for w, a in zip(weights, angles_deg))
        return math.degrees(math.atan2(ss, cs)) % 360.0

    MIN_HM0_FT = 0.2
    components: list[dict] = []

    for rank, pk in enumerate(merged):
        part = _partition_bins(rank)
        w       = [energy[i] * bw(i) for i in part]   # energy per bin (m²)
        total_e = sum(w)
        if total_e <= 0:
            continue

        hm0_ft = m_to_ft(4.0 * math.sqrt(total_e))
        Tp     = 1.0 / freqs[pk]
        if hm0_ft < MIN_HM0_FT or Tp < MIN_SWELL_PERIOD_S:
            continue

        # Peak-bin direction (what Surfline prints); energy-weighted circular
        # mean only when the peak bin carries the 999 fill.
        if dirs[pk] is not None:
            mean_dir = dirs[pk]
        else:
            known = [(wi, dirs[i]) for wi, i in zip(w, part) if dirs[i] is not None]
            if not known:
                continue
            mean_dir = _circular_mean([k[0] for k in known], [k[1] for k in known])

        # H² × T — partition energy proxy. Used both for component sort
        # below and as `energy` consumed by the buoy modal's max-single-
        # swell callout, the modal's energy-history chart, and the
        # spectrum chart. Same convention as _parse / fetch_buoy_history
        # and the wave models (wave_common), so partition #1 is picked the
        # same way on both sides of the CSC2 comparison.
        components.append({
            "height_ft":     round(hm0_ft, 2),
            "period_s":      round(Tp, 1),
            "direction_deg": round(mean_dir) % 360,
            "energy":        round(hm0_ft ** 2 * Tp, 1),
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

    Returns 0–1 items — the primary swell only (wind sea is intentionally
    excluded), dropped when its period is under MIN_SWELL_PERIOD_S (the same
    floor as _spectral_components and the wave models).
    """
    headers, data_lines = _split_ndbc(text)
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
        if not p or p < MIN_SWELL_PERIOD_S:   # wind chop, skip
            return
        h_ft   = m_to_ft(h_m)
        energy = round(h_ft ** 2 * p, 1) if (h_ft and p) else None  # H² × T convention
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
        bins = _parse_bins(parts, offset)
        if bins:
            result[ts.isoformat()] = bins
    return result


SPECTRAL_MATCH_TOL = timedelta(minutes=30)
# Inclusive: buoys logging stdmet every 10 min but spectra every 30 min leave
# the newest obs exactly 30 min past the last spectrum. NOAA-owned buoys
# publish stdmet and spectra on the same minute mark, but UCONN/USACE/UNH
# buoys (44091/44097/44098) report stdmet at :26/:56 while their spectra
# land on the hour, so exact-string matching never hits.


def _nearest_spectral_key(spec_times: list[tuple[datetime, str]], ts: datetime) -> str | None:
    """Key of the spectral row nearest `ts` within SPECTRAL_MATCH_TOL, or None.
    `spec_times` is [(datetime, iso_key), …] sorted ascending. The ONE pairing
    rule for stdmet ↔ spectrum, used by fetch_buoy (the BUOY NOW cell and the
    CSC2 obs logger) and fetch_buoy_history (the modal / historical strip) so
    the two can never show different decompositions for the same observation."""
    if not spec_times:
        return None
    dts = [dt for dt, _ in spec_times]
    i = bisect.bisect_left(dts, ts)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(dts):
            d = abs(dts[j] - ts)
            if d <= SPECTRAL_MATCH_TOL and (best is None or d < best[0]):
                best = (d, spec_times[j][1])
    return best[1] if best else None


def _fetch_historical_spectral(station_id: str, cutoff_dt: datetime) -> dict:
    """
    Fetch .data_spec + .swdir for all available rows and compute spectral
    components for each timestamp after cutoff_dt. Also return the raw
    energy+direction bins per timestamp for the buoy-modal spectrum graph.

    Returns {iso_timestamp: {
        "components": [component_dicts],
        "spectrum":   [[freq, energy_density, direction_deg_or_None], ...],
    }, ...}.

    A timestamp is included if raw spectrum bins are available, even if no
    swell-band components clear the Hm0 / period filter in _spectral_components.
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
    for ts_key, spec_bins in spec_all.items():
        if ts_key < cutoff_iso:
            continue
        swdir_bins = swdir_all.get(ts_key)
        # Build a freq→dir lookup for the spectrum bins; missing dirs → None.
        # NDBC emits 999.0 as the swdir fill marker.
        dir_by_freq = {f: (None if d in _FILL else d) for f, d in swdir_bins} if swdir_bins else {}
        spectrum = [
            [round(f, 5),
             round(e, 6) if e is not None else None,
             (round(dir_by_freq[f], 1) if dir_by_freq.get(f) is not None else None)]
            for f, e in spec_bins
        ]
        comps = _spectral_components(spec_bins, swdir_bins) if swdir_bins else []
        result[ts_key] = {"components": comps, "spectrum": spectrum}
    return result


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_buoy_history(station_id: str, days: int = 10) -> dict | None:
    """
    Fetch the last `days` of hourly buoy observations from NDBC realtime2,
    plus spectral swell components AND raw energy/direction bins for each
    timestamp where spectral data exists.

    Energy is height_ft² × period_s — the energy proxy shared with _parse,
    _spectral_components and the wave models (wave_common).
    """
    url = NDBC_URL.format(station_id=station_id)
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"[buoy_history] {station_id}: fetch failed — {type(e).__name__}: {e}")
        return None

    headers, data_lines = _split_ndbc(r.text)
    if not headers or not data_lines:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    records = []
    for line in data_lines:
        parts = line.split()
        row = dict(zip(headers, parts))
        try:
            ts = _row_ts(row)
        except Exception:
            continue
        if ts < cutoff:
            break  # file is newest-first

        wvht_m = _safe(row.get("WVHT"))
        dpd    = _safe(row.get("DPD"))
        mwd    = _safe_dir(row.get("MWD"))
        wvht_ft = m_to_ft(wvht_m)

        energy = round(wvht_ft ** 2 * dpd, 1) if (wvht_ft and dpd) else None

        records.append({
            "timestamp":     ts.isoformat(),
            "wave_height_ft": wvht_ft,
            "wave_period_s":  dpd,
            "wave_direction_deg": mwd,
            "energy":        energy,
            "components":    [],  # filled below if spectral data available
        })

    records.reverse()  # oldest-first for charting

    # Merge spectral components + raw spectrum bins, pairing each stdmet row
    # with the nearest spectrum (_nearest_spectral_key — shared with fetch_buoy).
    try:
        spec_map = _fetch_historical_spectral(station_id, cutoff)
        spec_dts = sorted(
            (datetime.fromisoformat(k), k) for k in spec_map.keys()
        )
        for rec in records:
            key = _nearest_spectral_key(spec_dts, datetime.fromisoformat(rec["timestamp"]))
            if key is None:
                continue
            entry = spec_map[key]
            if entry.get("components"):
                rec["components"] = entry["components"]
            if entry.get("spectrum"):
                rec["spectrum"] = entry["spectrum"]
    except Exception as e:
        print(f"[buoy_history] {station_id}: spectral merge error — {type(e).__name__}: {e}")

    record_api_calls("NOAA_buoy_history", 1)

    return {"station_id": station_id, "records": records}


@ttl_cache(ttl_seconds=600, skip_none=True)
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
        if wvht is None:
            # Entire file had no valid WVHT — try the other URL
            print(f"[buoy] {station_id} ({src}): all rows MM for wave height, trying next URL")
            continue
        print(f"[buoy] {station_id} ({src}): OK — {wvht}ft @ {result.get('wave_period_s')}s")

        # ── Fetch individual swell components ─────────────────────────────
        # Preferred: raw spectral files (.data_spec + .swdir) → Surfline-equivalent,
        #            the row nearest this stdmet row's timestamp. NDBC posts the
        #            two files minutes apart, so "first line of each" could pair
        #            a 17:20 stdmet reading with a 17:50 spectrum while the
        #            history path paired it with 17:20 — cell ≠ modal.
        # Fallback:  spectral summary (.spec) → 1 swell only
        comps: list = []
        try:
            rds = requests.get(NDBC_DATA_SPEC_URL.format(station_id=station_id),
                               timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
            rsw = requests.get(NDBC_SWDIR_URL.format(station_id=station_id),
                               timeout=15, headers={"User-Agent": "ColeSurfs/1.0"})
            if rds.status_code == 200 and rsw.status_code == 200:
                spec_all  = _parse_spectral_file_all_rows(rds.text, value_offset=1)
                swdir_all = _parse_spectral_file_all_rows(rsw.text, value_offset=0)
                spec_dts  = sorted((datetime.fromisoformat(k), k)
                                   for k in spec_all if k in swdir_all)
                key = None
                if result.get("timestamp"):
                    key = _nearest_spectral_key(spec_dts, datetime.fromisoformat(result["timestamp"]))
                if key is not None:
                    comps = _spectral_components(spec_all[key], swdir_all[key])
                    print(f"[buoy] {station_id} spectral: {len(comps)} component(s) "
                          f"(data_spec+swdir @ {key})")
                elif spec_dts:
                    print(f"[buoy] {station_id} no spectrum within "
                          f"{SPECTRAL_MATCH_TOL} of {result.get('timestamp')}, trying .spec")
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


# ─── Historical context (obs + model-agreement indicator) ─────────────────────
# Data source: fetch_buoy_history (already cached) for records, plus local
# .csc2_data/forecasts/ parquets for per-hour EURO/GFS categories. No extra
# NDBC traffic.

def _hour_iso_z(ts_iso: str) -> str | None:
    """Snap an observation timestamp to the nearest top-of-hour and return
    the forecast-parquet `valid_utc` string form. NDBC obs land every 10 min;
    forecasts are hourly, so we nearest-hour round for the lookup."""
    if not ts_iso:
        return None
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    floor = dt.replace(minute=0, second=0, microsecond=0)
    if (dt - floor) >= timedelta(minutes=30):
        floor = floor + timedelta(hours=1)
    return floor.strftime("%Y-%m-%dT%H:%M:%SZ")


def _primary_height_period(rec: dict) -> tuple[float | None, float | None]:
    """Primary swell Hs/Tp for a history record: components[0] if present,
    else the combined wave_height_ft / wave_period_s."""
    comps = rec.get("components") or []
    if comps:
        c = comps[0]
        return c.get("height_ft"), c.get("period_s")
    return rec.get("wave_height_ft"), rec.get("wave_period_s")


def _load_model_row_map(station_id: str, model: str, months: set[tuple[int, int]]) -> dict:
    """Read the relevant cycle parquets for one model and build a
    {valid_utc_z: (sw1_height_ft, sw1_period_s)} map keyed by the most-recent
    cycle_utc ≤ valid_utc. Missing month folders or unreadable shards are
    silently skipped.
    """
    import pandas as pd  # deferred: csc2 already installs pandas
    from csc2.schema import FORECASTS_DIR

    buoy_dir = FORECASTS_DIR / f"model={model}" / f"buoy={station_id}"
    frames = []
    for (y, m) in months:
        mdir = buoy_dir / f"year={y}" / f"month={m:02d}"
        if not mdir.exists():
            continue
        for p in sorted(mdir.glob("cycle=*.parquet")):
            try:
                df = pd.read_parquet(p, columns=[
                    "valid_utc", "cycle_utc", "sw1_height_ft", "sw1_period_s",
                ])
            except Exception:
                continue
            frames.append(df)
    if not frames:
        return {}

    try:
        allrows = pd.concat(frames, ignore_index=True)
        # Pick the row with the latest cycle_utc for each valid_utc
        allrows = allrows.sort_values("cycle_utc").drop_duplicates(
            subset=["valid_utc"], keep="last"
        )
    except Exception:
        return {}

    out = {}
    for _, r in allrows.iterrows():
        h = r.get("sw1_height_ft")
        p = r.get("sw1_period_s")
        # pandas NaN → None
        h = None if (h is None or (isinstance(h, float) and math.isnan(h))) else float(h)
        p = None if (p is None or (isinstance(p, float) and math.isnan(p))) else float(p)
        out[str(r["valid_utc"])] = (h, p)
    return out


def _months_spanned(records: list[dict]) -> set[tuple[int, int]]:
    months: set[tuple[int, int]] = set()
    for rec in records:
        ts = rec.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        months.add((dt.year, dt.month))
    return months


@ttl_cache(ttl_seconds=1800, skip_none=True)
def fetch_buoy_historical_context(station_id: str, days: int = 10) -> dict | None:
    """
    Return observed buoy history enriched with a `model_agreement` field per
    record (true | false | null). Null when either the buoy isn't in CSC2's
    archive scope or no forecast row covers that hour.
    """
    try:
        from csc2.schema import BUOY_IDS
    except Exception:
        BUOY_IDS = []  # csc2 unavailable → all records get model_agreement=null

    base = fetch_buoy_history(station_id, days=days)
    if base is None:
        return None

    records = base.get("records") or []
    out_records: list[dict] = []

    has_archive = station_id in BUOY_IDS
    euro_map: dict = {}
    gfs_map: dict = {}
    if has_archive and records:
        months = _months_spanned(records)
        try:
            euro_map = _load_model_row_map(station_id, "EURO", months)
        except Exception as e:
            print(f"[buoy_historical_context] EURO load failed — {type(e).__name__}: {e}")
            euro_map = {}
        try:
            gfs_map = _load_model_row_map(station_id, "GFS", months)
        except Exception as e:
            print(f"[buoy_historical_context] GFS load failed — {type(e).__name__}: {e}")
            gfs_map = {}

    for rec in records:
        h_obs, p_obs = _primary_height_period(rec)
        obs_cat = swell_rules.categorize(h_obs, p_obs) if (h_obs and p_obs) else None

        agreement = None
        if has_archive and obs_cat:
            vkey = _hour_iso_z(rec.get("timestamp"))
            euro = euro_map.get(vkey) if vkey else None
            gfs  = gfs_map.get(vkey) if vkey else None
            if euro and gfs and euro[0] is not None and euro[1] is not None \
               and gfs[0] is not None and gfs[1] is not None:
                euro_cat = swell_rules.categorize(euro[0], euro[1])
                gfs_cat  = swell_rules.categorize(gfs[0], gfs[1])
                agreement = (euro_cat == obs_cat) and (gfs_cat == obs_cat)

        out = {
            "timestamp":          rec.get("timestamp"),
            "wave_height_ft":     rec.get("wave_height_ft"),
            "wave_period_s":      rec.get("wave_period_s"),
            "wave_direction_deg": rec.get("wave_direction_deg"),
            "components":         rec.get("components") or [],
            "observed_cat":       obs_cat,
            "model_agreement":    agreement,
        }
        out_records.append(out)

    return {"station_id": station_id, "records": out_records}
