"""CSC v2 backfill — historical primary-swell decomposition.

For each CSC buoy, gather spectral wave observations from four sources,
decompose each timestamp into swell partitions using the same algorithm
that `buoy.py` runs on live fetches, and write Parquet shards to
`.csc_data/primary_swell/buoy=<id>/year=<y>/`.

Sources, in priority order:
  1. NDBC THREDDS OPeNDAP (`ndbc_thredds_hist`) — PRIMARY. One
     accumulating `.nc` per station under dods.ndbc.noaa.gov, carrying
     the full multi-year spectral record (e.g. 44097 → ~72k obs back to
     2017-11). Biggest single source.
  2. NDBC historical `.swden` / `.swdir` yearly gzipped text archives
     (`ndbc_swden_hist`) — augmenter for any year-gaps in THREDDS.
  3. NDBC realtime2 rolling 45-day spectral (`ndbc_realtime45d`) —
     bootstraps the tail end for stations lacking both archives.
  4. CDIP THREDDS historic.nc (`cdip_thredds`) — for the two
     CDIP-operated buoys (46221, 46268) not in NDBC's swden catalog.
     Reads the raw 2D spectral tables (waveEnergyDensity +
     waveMeanDirection) and pipes them through the same
     `_spectral_components` decomposer used by the live dashboard, so the
     training label and the dashboard's buoy-components row agree
     bit-for-bit (modulo the buoy's own measurement grid).

This module intentionally *reuses* `buoy._spectral_components` so the
historical primary-swell values are bit-for-bit consistent with what the
live observation logger produces — which means the retrained CSC sees a
single uniform target at train and inference time.

Run:
    python -m csc.backfill_primary_swell                   # all 8 buoys
    python -m csc.backfill_primary_swell --buoy 44097      # one buoy
    python -m csc.backfill_primary_swell --years 2023 2024 # override window

Output shape (per row):
    buoy_id, valid_utc, partition(1 or 2), hm0_m, tm_s, dir_deg,
    energy, source
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

from csc.schema import BUOYS, CSC_DATA_DIR, ensure_dirs

# Reuse the live-path spectral decomposer — identical to buoy.py's component list.
# buoy.py lives at the project root (parent of csc/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from buoy import _spectral_components  # noqa: E402

PRIMARY_SWELL_DIR = CSC_DATA_DIR / "primary_swell"

NDBC_SWDEN_HIST = (
    "https://www.ndbc.noaa.gov/data/historical/swden/{station}w{year}.txt.gz"
)
NDBC_SWDIR_HIST = (
    "https://www.ndbc.noaa.gov/data/historical/swdir/{station}d{year}.txt.gz"
)

FT_PER_M = 3.28084


# ─── Historical spectral file parsing ─────────────────────────────────────


def _download_gz(url: str, timeout_s: float = 90.0) -> str | None:
    """Fetch a gzip-compressed NDBC archive; return decoded text or None on 404."""
    try:
        r = requests.get(url, timeout=timeout_s,
                         headers={"User-Agent": "ColeSurfs/1.0"})
    except requests.RequestException as e:
        print(f"  [!] fetch error {url}: {e}", file=sys.stderr)
        return None
    if r.status_code == 404:
        return None
    if not r.ok:
        print(f"  [!] HTTP {r.status_code} {url}", file=sys.stderr)
        return None
    try:
        return gzip.decompress(r.content).decode("utf-8", errors="replace")
    except OSError as e:
        # Some early archive files aren't gzipped — try raw.
        try:
            return r.content.decode("utf-8", errors="replace")
        except Exception:
            print(f"  [!] gunzip failed {url}: {e}", file=sys.stderr)
            return None


def _iter_spectral_rows(text: str) -> Iterator[tuple[datetime, list[tuple[float, float]]]]:
    """Iterate (timestamp, [(freq, value), ...]) from a historical NDBC swden
    or swdir archive. Both files share the same row layout:

      #YY MM DD hh [mm]  v1 v2 ... vN
      #yr mo dy hr [mn]   .0200 .0325 ... .4850   ← units row; frequencies are
                                                   the column headers, not
                                                   inline (val,freq) pairs
                                                   like the realtime .data_spec
                                                   format.

    Historical files use column-headered frequencies: the 2nd comment row
    lists frequency labels beginning at the first value column. We parse
    both the header frequencies and each data row's values, zipping them.

    Some older files omit `mm` — we detect column count against header
    length to auto-adapt.
    """
    lines = text.splitlines()
    header: list[str] | None = None
    freq_header: list[float] | None = None
    # First commented line is column names (YY MM DD hh [mm] plus
    # frequency labels like ".0200"). Only swden/swdir use this schema.
    for line in lines:
        if line.startswith("#"):
            if header is None:
                header = line.lstrip("# ").split()
                # Frequency columns are whatever floats follow the time cols.
                freq_tokens: list[float] = []
                for tok in header:
                    try:
                        freq_tokens.append(float(tok))
                    except ValueError:
                        continue
                if freq_tokens:
                    freq_header = freq_tokens
            continue
        if not line.strip():
            continue
        parts = line.split()
        # Layout: YY MM DD hh [mm] v1 v2 ... vN. Try 5 time cols first
        # (post-2005 format), fall back to 4.
        for n_time_cols in (5, 4):
            if freq_header is None:
                return
            if len(parts) < n_time_cols + len(freq_header):
                continue
            try:
                yr = int(parts[0])
                if yr < 100:
                    yr += 1900 if yr > 50 else 2000
                mo = int(parts[1]); dy = int(parts[2]); hr = int(parts[3])
                mn = int(parts[4]) if n_time_cols == 5 else 0
                ts = datetime(yr, mo, dy, hr, mn, tzinfo=timezone.utc)
            except (ValueError, IndexError):
                continue
            vals: list[tuple[float, float]] = []
            for i, f in enumerate(freq_header):
                try:
                    v = float(parts[n_time_cols + i])
                except (ValueError, IndexError):
                    continue
                vals.append((f, v))
            if vals:
                yield ts, vals
            break


def _fetch_buoy_year(station: str, year: int, sleep_s: float = 0.5
                     ) -> tuple[dict[datetime, list], dict[datetime, list]]:
    """Return ({ts: [(f,e),...]}, {ts: [(f,dir),...]}) for one (buoy, year)."""
    spec_text = _download_gz(NDBC_SWDEN_HIST.format(station=station, year=year))
    time.sleep(sleep_s)
    dir_text = _download_gz(NDBC_SWDIR_HIST.format(station=station, year=year))
    time.sleep(sleep_s)
    spec: dict[datetime, list[tuple[float, float]]] = {}
    dirs: dict[datetime, list[tuple[float, float]]] = {}
    if spec_text:
        for ts, bins in _iter_spectral_rows(spec_text):
            spec[ts] = bins
    if dir_text:
        for ts, bins in _iter_spectral_rows(dir_text):
            dirs[ts] = bins
    return spec, dirs


def _decompose_year(station: str, year: int, sleep_s: float = 0.5
                    ) -> list[dict[str, Any]]:
    """Download + decompose one (buoy, year). Returns long rows — one per
    (timestamp × partition)."""
    spec, dirs = _fetch_buoy_year(station, year, sleep_s=sleep_s)
    if not spec:
        return []
    rows: list[dict[str, Any]] = []
    ingest_s = datetime.now(timezone.utc).isoformat()
    for ts, spec_bins in spec.items():
        dir_bins = dirs.get(ts)
        if dir_bins is None:
            # Some timestamps in swden may lack a matching swdir entry.
            continue
        comps = _spectral_components(spec_bins, dir_bins)
        if not comps:
            continue
        for i, c in enumerate(comps, start=1):
            h_ft = c.get("height_ft")
            tm = c.get("period_s")
            dp = c.get("direction_deg")
            energy = c.get("energy")
            if h_ft is None or tm is None or dp is None:
                continue
            rows.append({
                "buoy_id": station,
                "valid_utc": ts.isoformat(),
                "partition": int(i),
                "hm0_m": float(h_ft) / FT_PER_M,
                "tm_s": float(tm),
                "dir_deg": float(dp),
                "energy": float(energy) if energy is not None else None,
                "source": "ndbc_swden_hist",
                "ingest_utc": ingest_s,
            })
    return rows


# ─── Source 2: NDBC realtime2 rolling 45-day spectral ────────────────────
# Every active NDBC buoy serves its last ~45 days of spectral obs in the
# realtime2 .data_spec + .swdir files, regardless of whether the yearly
# historical .swden/.swdir archive exists. Pull these for EVERY buoy to
# bootstrap training even for stations NDBC doesn't archive historically.


def _decompose_realtime_45d(station: str) -> list[dict[str, Any]]:
    """Call buoy._fetch_historical_spectral which parses all rows of the
    live 45-day realtime2 .data_spec + .swdir and returns
    `{iso_ts: [component_dicts]}`."""
    from datetime import timedelta
    try:
        from buoy import _fetch_historical_spectral
    except Exception as e:
        print(f"  [!] {station}: cannot import buoy._fetch_historical_spectral: {e}")
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=60)
    try:
        by_ts = _fetch_historical_spectral(station, cutoff)
    except Exception:
        traceback.print_exc()
        return []
    if not by_ts:
        return []
    rows: list[dict[str, Any]] = []
    ingest_s = datetime.now(timezone.utc).isoformat()
    for ts_iso, comps in by_ts.items():
        for i, c in enumerate(comps, start=1):
            h_ft = c.get("height_ft")
            tm = c.get("period_s")
            dp = c.get("direction_deg")
            energy = c.get("energy")
            if h_ft is None or tm is None or dp is None:
                continue
            rows.append({
                "buoy_id": station,
                "valid_utc": ts_iso,
                "partition": int(i),
                "hm0_m": float(h_ft) / FT_PER_M,
                "tm_s": float(tm),
                "dir_deg": float(dp),
                "energy": float(energy) if energy is not None else None,
                "source": "ndbc_realtime45d",
                "ingest_utc": ingest_s,
            })
    return rows


# ─── Source 3: NDBC THREDDS per-station accumulating spectral NetCDF ─────
# Every NDBC spectral station ships one OPeNDAP-servable NetCDF on
# dods.ndbc.noaa.gov that contains its full spectral record (often years).
# This is the largest single source for CSC training — e.g. station 44097
# (Block Island Sound) carries ~72k observations back to 2017-11. Unlike
# the yearly `.swden` / `.swdir` text archives, a single `.nc` holds both
# the energy density and the α1 direction, so no join is required.

NDBC_THREDDS_CATALOG = (
    "https://dods.ndbc.noaa.gov/thredds/catalog/data/swden/{station}/catalog.xml"
)
NDBC_THREDDS_OPENDAP = (
    "https://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/{station}/{filename}"
)


def _list_ndbc_thredds_files(station: str, timeout_s: float = 30.0
                              ) -> list[str]:
    """Return the list of `.nc` filenames exposed under the THREDDS catalog
    for this station. Handles both the w9999 (active accumulating) and
    per-year historical files without hard-coding either suffix."""
    import re
    url = NDBC_THREDDS_CATALOG.format(station=station)
    try:
        r = requests.get(url, timeout=timeout_s,
                         headers={"User-Agent": "ColeSurfs/1.0"})
    except requests.RequestException as e:
        print(f"  [!] {station}: THREDDS catalog fetch failed: {e}", file=sys.stderr)
        return []
    if not r.ok:
        return []
    # THREDDS catalog.xml lists datasets as `<dataset name="..." urlPath="..."/>`.
    # Pull urlPath values ending in .nc — they're station-relative filenames.
    names: list[str] = []
    for m in re.finditer(r'urlPath="([^"]+\.nc)"', r.text):
        path = m.group(1)
        names.append(path.rsplit("/", 1)[-1])
    # Fall back to dataset name attributes if urlPath didn't match.
    if not names:
        for m in re.finditer(r'name="([^"]+\.nc)"', r.text):
            names.append(m.group(1))
    # De-dupe while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _decompose_ndbc_thredds(station: str, years: list[int]
                             ) -> list[dict[str, Any]]:
    """Walk NDBC THREDDS for `station`, stream each `.nc` over OPeNDAP, and
    decompose every observation whose timestamp falls in `years`. Reuses
    `buoy._spectral_components` for bit-parity with the live logger.

    Fill-value handling mirrors the NDBC CF conventions: energy == 999.0
    and direction >= 990 are treated as missing. Timestamps where no
    finite swell-band bin survives are dropped. Reads are chunked by year
    so a 72k-row file doesn't require pulling the whole array at once.
    """
    try:
        import netCDF4  # type: ignore
    except ImportError:
        print(f"  [!] {station}: netCDF4 not installed — skipping THREDDS")
        return []
    import numpy as np

    filenames = _list_ndbc_thredds_files(station)
    if not filenames:
        print(f"  [!] {station}: no NetCDF files in THREDDS catalog")
        return []

    yrs_set = set(years)
    rows: list[dict[str, Any]] = []
    ingest_s = datetime.now(timezone.utc).isoformat()

    for fname in filenames:
        url = NDBC_THREDDS_OPENDAP.format(station=station, filename=fname)
        print(f"  {station} → THREDDS OPeNDAP: {fname}")
        try:
            ds = netCDF4.Dataset(url, "r")
        except Exception as e:
            print(f"  [!] {station}: cannot open {fname}: {e}", file=sys.stderr)
            continue
        try:
            time_var = ds.variables.get("time")
            freq_var = ds.variables.get("frequency")
            swd_var = ds.variables.get("spectral_wave_density")
            dir_var = ds.variables.get("mean_wave_dir")
            if (time_var is None or freq_var is None
                    or swd_var is None or dir_var is None):
                print(f"  [!] {station}: {fname} missing expected variables")
                continue

            # Read time + frequency headers in full — small arrays.
            times = np.asarray(time_var[:], dtype="int64")
            freqs = np.asarray(freq_var[:], dtype="float64")
            n_time = len(times)
            n_freq = len(freqs)
            if n_time == 0 or n_freq == 0:
                continue

            # Group time indices by calendar year so we slice the big
            # arrays in year-chunks instead of loading everything.
            idx_years = np.array(
                [datetime.fromtimestamp(int(t), tz=timezone.utc).year
                 for t in times],
                dtype="int32",
            )
            for y in sorted({int(yy) for yy in idx_years.tolist()}):
                if y not in yrs_set:
                    continue
                mask = np.where(idx_years == y)[0]
                if mask.size == 0:
                    continue
                i0, i1 = int(mask.min()), int(mask.max()) + 1
                # Slice spec + dir for the year-block. Shape is
                # [time, frequency, lat=1, lon=1]; squeeze trailing axes.
                try:
                    swd_block = np.asarray(swd_var[i0:i1, :, :, :],
                                            dtype="float64")
                    dir_block = np.asarray(dir_var[i0:i1, :, :, :],
                                            dtype="float64")
                except Exception as e:
                    print(f"  [!] {station}: OPeNDAP slice {y} failed: {e}",
                          file=sys.stderr)
                    continue
                swd_block = np.squeeze(swd_block)
                dir_block = np.squeeze(dir_block)
                if swd_block.ndim == 1:
                    swd_block = swd_block.reshape(1, -1)
                if dir_block.ndim == 1:
                    dir_block = dir_block.reshape(1, -1)

                n_block = swd_block.shape[0]
                emitted = 0
                for k in range(n_block):
                    global_i = i0 + k
                    ts = datetime.fromtimestamp(
                        int(times[global_i]), tz=timezone.utc)
                    if ts.year != y:
                        continue
                    e_row = swd_block[k]
                    d_row = dir_block[k]
                    # Feed ALL frequency bins to _spectral_components to
                    # preserve bin-width geometry (bw(i) uses freqs[i±1]).
                    # The live text path never drops bins — it keeps 999.0
                    # fills inline. Dropping them here would shorten the
                    # freqs list and widen Δf for the surviving bins,
                    # changing Hm0 = 4·√Σ E Δf. Instead, substitute fill /
                    # NaN energies with 0.0 (below NOISE_FLOOR, so they
                    # cannot become peaks and contribute zero energy to any
                    # partition integral). Direction value for a
                    # zero-energy bin is irrelevant because the circular
                    # mean is energy-weighted.
                    #
                    # Fill-value conventions per NDBC CF: 999.0 for energy,
                    # 999 for direction. NaN can also appear after
                    # netCDF4 auto-masking.
                    spec_bins: list[tuple[float, float]] = []
                    dir_bins: list[tuple[float, float]] = []
                    finite_count = 0
                    for j in range(n_freq):
                        f_val = float(freqs[j])
                        e_val = float(e_row[j])
                        d_val = float(d_row[j])
                        if not np.isfinite(e_val) or e_val >= 999.0:
                            e_val = 0.0
                            d_val = 0.0
                        else:
                            finite_count += 1
                            if not np.isfinite(d_val) or d_val >= 990.0:
                                d_val = 0.0
                        spec_bins.append((f_val, e_val))
                        dir_bins.append((f_val, d_val))
                    # Drop timestamp if too few finite energy bins survive.
                    if finite_count < 8:
                        continue
                    try:
                        comps = _spectral_components(spec_bins, dir_bins)
                    except Exception:
                        continue
                    if not comps:
                        continue
                    for i, c in enumerate(comps, start=1):
                        h_ft = c.get("height_ft")
                        tm = c.get("period_s")
                        dp = c.get("direction_deg")
                        energy = c.get("energy")
                        if h_ft is None or tm is None or dp is None:
                            continue
                        rows.append({
                            "buoy_id": station,
                            "valid_utc": ts.isoformat(),
                            "partition": int(i),
                            "hm0_m": float(h_ft) / FT_PER_M,
                            "tm_s": float(tm),
                            "dir_deg": float(dp),
                            "energy": float(energy) if energy is not None else None,
                            "source": "ndbc_thredds_hist",
                            "ingest_utc": ingest_s,
                        })
                        emitted += 1
                print(f"    {station} {y}: {emitted:>7} component rows "
                      f"from {n_block} obs")
        finally:
            try:
                ds.close()
            except Exception:
                pass

    return rows


# ─── Source 4: CDIP THREDDS for 46221 (028) and 46268 (266) ──────────────


CDIP_STATIONS = {
    "46221": "028",   # Santa Monica Bay — CDIP 028
    "46268": "266",   # Topanga — CDIP 266
}
CDIP_HISTORIC_URL = (
    "https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/"
    "{cdip}p1/{cdip}p1_historic.nc"
)


def _decompose_cdip(station: str, years: list[int]) -> list[dict[str, Any]]:
    """Pull CDIP historic.nc for the mapped station and emit primary-swell
    rows using the SAME spectral decomposer that `buoy.py` runs on live
    NDBC realtime2 pulls — so the historical label and the dashboard's
    buoy-components row agree on a bit-for-bit basis at 46221 and 46268.

    Inputs read from the CDIP p1_historic.nc file:
      waveTime[time]                 UTC seconds
      waveFrequency[frequency]       Hz
      waveEnergyDensity[time, freq]  m² Hz⁻¹   (== NDBC spectral_wave_density)
      waveMeanDirection[time, freq]  deg true, FROM
                                       (== NDBC mean_wave_dir α1)

    When waveMeanDirection is absent we fall back to the directional
    Fourier coefficients:
      alpha1(f) = (atan2(-b1, -a1)) mod 360, in degrees     (FROM convention)
    CDIP stores waveA1Value and waveB1Value with the TO-wave convention; the
    sign flip yields the FROM direction NDBC uses.

    The summary-param row (Hs, Tp, Dp) used previously produced
    dashboard-divergent training labels because `waveHs` is a full-spectrum
    significant wave height (not a primary-partition Hm0) and `waveTp` is
    the peak period (not the energy-weighted mean Tm that
    `_spectral_components` returns). This function now matches the live
    path exactly, modulo the input grid."""
    cdip = CDIP_STATIONS.get(station)
    if not cdip:
        return []
    try:
        import netCDF4  # type: ignore
    except ImportError:
        print(f"  [!] {station}: netCDF4 not installed — skipping CDIP")
        return []
    import math
    import numpy as np
    import tempfile, urllib.request
    url = CDIP_HISTORIC_URL.format(cdip=cdip)
    print(f"  {station} → CDIP {cdip}: downloading historic.nc (large)")
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            # Stream to temp file to avoid loading everything in RAM
            with urllib.request.urlopen(url, timeout=600) as resp:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
            tmp_path = tmp.name
    except Exception as e:
        print(f"  [!] {station}: CDIP download failed: {e}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        ds = netCDF4.Dataset(tmp_path, "r")
        time_var = ds.variables.get("waveTime")
        freq_var = ds.variables.get("waveFrequency")
        ed_var = ds.variables.get("waveEnergyDensity")
        mdir_var = ds.variables.get("waveMeanDirection")
        a1_var = ds.variables.get("waveA1Value")
        b1_var = ds.variables.get("waveB1Value")
        if time_var is None or freq_var is None or ed_var is None:
            print(f"  [!] {station}: CDIP file missing waveTime/Frequency/"
                  "EnergyDensity — skipping")
            return []
        if mdir_var is None and (a1_var is None or b1_var is None):
            print(f"  [!] {station}: CDIP file missing both waveMeanDirection "
                  "and waveA1Value/waveB1Value — cannot decompose")
            return []
        freqs = np.asarray(freq_var[:], dtype="float64")
        n_freq = len(freqs)
        n_time = time_var.shape[0]
        if n_time == 0 or n_freq == 0:
            return []
        yrs_set = set(years)
        ingest_s = datetime.now(timezone.utc).isoformat()
        # Chunk by calendar year to avoid pulling the full time × freq array
        # at once (historic.nc can be 1+ GB).
        times = np.asarray(time_var[:], dtype="int64")
        idx_years = np.array(
            [datetime.fromtimestamp(int(t), tz=timezone.utc).year
             for t in times],
            dtype="int32",
        )
        for y in sorted({int(yy) for yy in idx_years.tolist()}):
            if y not in yrs_set:
                continue
            mask = np.where(idx_years == y)[0]
            if mask.size == 0:
                continue
            i0, i1 = int(mask.min()), int(mask.max()) + 1
            try:
                ed_block = np.asarray(ed_var[i0:i1, :], dtype="float64")
                if mdir_var is not None:
                    dir_block = np.asarray(mdir_var[i0:i1, :],
                                           dtype="float64")
                else:
                    a1_block = np.asarray(a1_var[i0:i1, :], dtype="float64")
                    b1_block = np.asarray(b1_var[i0:i1, :], dtype="float64")
                    # CDIP a1/b1 are the TO-wave Fourier moments; flip sign
                    # for FROM convention matching NDBC α1.
                    dir_block = (
                        np.degrees(np.arctan2(-b1_block, -a1_block)) % 360.0
                    )
            except Exception as e:
                print(f"  [!] {station}: CDIP slice {y} failed: {e}",
                      file=sys.stderr)
                continue
            n_block = ed_block.shape[0]
            emitted = 0
            for k in range(n_block):
                global_i = i0 + k
                ts = datetime.fromtimestamp(int(times[global_i]),
                                            tz=timezone.utc)
                if ts.year != y:
                    continue
                e_row = ed_block[k]
                d_row = dir_block[k]
                spec_bins: list[tuple[float, float]] = []
                dir_bins: list[tuple[float, float]] = []
                for j in range(n_freq):
                    e_val = float(e_row[j])
                    d_val = float(d_row[j])
                    # CDIP fill values: typically -999 or NaN after
                    # auto-masking. Also reject absurd positives (>= 999)
                    # mirroring the NDBC THREDDS path.
                    if not math.isfinite(e_val) or e_val < 0 or e_val >= 999.0:
                        continue
                    if not math.isfinite(d_val):
                        continue
                    f_val = float(freqs[j])
                    if not math.isfinite(f_val) or f_val <= 0:
                        continue
                    spec_bins.append((f_val, e_val))
                    dir_bins.append((f_val, d_val % 360.0))
                if len(spec_bins) < 8:
                    continue
                try:
                    comps = _spectral_components(spec_bins, dir_bins)
                except Exception:
                    continue
                if not comps:
                    continue
                for i, c in enumerate(comps, start=1):
                    h_ft = c.get("height_ft")
                    tm = c.get("period_s")
                    dp = c.get("direction_deg")
                    energy = c.get("energy")
                    if h_ft is None or tm is None or dp is None:
                        continue
                    rows.append({
                        "buoy_id": station,
                        "valid_utc": ts.isoformat(),
                        "partition": int(i),
                        "hm0_m": float(h_ft) / FT_PER_M,
                        "tm_s": float(tm),
                        "dir_deg": float(dp),
                        "energy": float(energy) if energy is not None else None,
                        "source": "cdip_thredds",
                        "ingest_utc": ingest_s,
                    })
                    emitted += 1
            print(f"    {station} {y}: {emitted:>7} component rows "
                  f"from {n_block} obs")
        ds.close()
    except Exception:
        traceback.print_exc()
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
    return rows


# ─── Output shard writer ──────────────────────────────────────────────────


def _shard_path(buoy_id: str, year: int) -> Path:
    return (PRIMARY_SWELL_DIR / f"buoy={buoy_id}" / f"year={year}"
            / "primary.parquet")


def _write_shard(buoy_id: str, year: int, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    path = _shard_path(buoy_id, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False, compression="snappy")
    return len(df)


# ─── Orchestration ────────────────────────────────────────────────────────


def _default_years() -> list[int]:
    # NDBC THREDDS coverage for the longest CSC station (44097) reaches
    # back to 2017-11, so default to that floor. Stations with shorter
    # records naturally emit zero rows for years outside their coverage.
    today = datetime.now(timezone.utc).date()
    return list(range(2017, today.year + 1))


def run_backfill(buoy_ids: list[str] | None = None,
                 years: list[int] | None = None,
                 sleep_s: float = 0.5) -> dict[str, Any]:
    """Fetch + decompose historical swden/swdir for each (buoy, year).

    Writes one Parquet shard per (buoy, year). Prints a coverage table at
    the end. Returns a structured report for programmatic consumers.
    """
    ensure_dirs()
    PRIMARY_SWELL_DIR.mkdir(parents=True, exist_ok=True)
    target_ids = buoy_ids or [b[0] for b in BUOYS]
    target_years = years or _default_years()

    coverage: dict[str, dict[int, int]] = {bid: {} for bid in target_ids}
    warnings: list[str] = []

    for bid in target_ids:
        meta = next((b for b in BUOYS if b[0] == bid), None)
        label = meta[1] if meta else "?"
        print(f"\n── {bid}  {label} ──", flush=True)
        had_any = False

        # Source 1 (PRIMARY): NDBC THREDDS accumulating spectral NetCDF.
        # This is the biggest single source — one `.nc` per station often
        # carries the full multi-year record (e.g. 44097 → 8.4 yrs). CDIP
        # stations (46221, 46268) aren't in this catalog and fall through
        # to empty cleanly.
        if bid not in CDIP_STATIONS:
            try:
                thr_rows = _decompose_ndbc_thredds(bid, target_years)
            except Exception:
                traceback.print_exc()
                thr_rows = []
            if thr_rows:
                by_year: dict[int, list] = {}
                for r in thr_rows:
                    y = int(r["valid_utc"][:4])
                    by_year.setdefault(y, []).append(r)
                for y, rs in by_year.items():
                    path = _shard_path(bid, y).with_name(
                        "primary_thredds.parquet")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rs).to_parquet(
                        path, index=False, compression="snappy")
                    coverage[bid][y] = coverage[bid].get(y, 0) + len(rs)
                    had_any = True
                    print(f"  {bid} {y} (ndbc_thredds_hist): {len(rs):>7} rows")

        # Source 2: NDBC historical swden/swdir yearly archive (404s for
        # many stations — try anyway). Augments THREDDS for years where
        # the accumulating NetCDF is thin or missing.
        for y in target_years:
            try:
                rows = _decompose_year(bid, y, sleep_s=sleep_s)
            except Exception:
                traceback.print_exc()
                rows = []
            n = _write_shard(bid, y, rows)
            coverage[bid][y] = coverage[bid].get(y, 0) + n
            if n > 0:
                had_any = True
            print(f"  {bid} {y} (ndbc_swden_hist): {n:>7} rows")

        # Source 3: NDBC realtime2 rolling 45-day spectral — ALL buoys.
        # Produces ~1,000-1,500 rows per buoy regardless of historical
        # archive availability. Write under the current-year shard keyed
        # by source so it doesn't collide with the historical pull.
        try:
            rt_rows = _decompose_realtime_45d(bid)
        except Exception:
            traceback.print_exc()
            rt_rows = []
        if rt_rows:
            # Partition rows across years in case the 45-day window crosses a
            # year boundary.
            by_year: dict[int, list] = {}
            for r in rt_rows:
                y = int(r["valid_utc"][:4])
                by_year.setdefault(y, []).append(r)
            for y, rs in by_year.items():
                path = _shard_path(bid, y).with_name("primary_rt45d.parquet")
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rs).to_parquet(path, index=False, compression="snappy")
                coverage[bid][y] = coverage[bid].get(y, 0) + len(rs)
                had_any = True
                print(f"  {bid} {y} (ndbc_realtime45d): {len(rs):>7} rows")

        # Source 4: CDIP THREDDS for the two CDIP-operated buoys.
        if bid in CDIP_STATIONS:
            try:
                cdip_rows = _decompose_cdip(bid, target_years)
            except Exception:
                traceback.print_exc()
                cdip_rows = []
            if cdip_rows:
                by_year: dict[int, list] = {}
                for r in cdip_rows:
                    y = int(r["valid_utc"][:4])
                    by_year.setdefault(y, []).append(r)
                for y, rs in by_year.items():
                    path = _shard_path(bid, y).with_name("primary_cdip.parquet")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rs).to_parquet(path, index=False, compression="snappy")
                    coverage[bid][y] = coverage[bid].get(y, 0) + len(rs)
                    had_any = True
                    print(f"  {bid} {y} (cdip_thredds): {len(rs):>7} rows")

        if not had_any:
            msg = (f"[warn] {bid}: no spectral data from any source — "
                   "training will have zero rows for this buoy")
            print(msg)
            warnings.append(msg)

    _print_coverage(coverage, target_years)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "buoys": target_ids,
        "years": target_years,
        "coverage": coverage,
        "warnings": warnings,
    }


def _print_coverage(coverage: dict[str, dict[int, int]],
                    years: list[int]) -> None:
    print("\n─────────── primary-swell backfill coverage ───────────")
    header = f"{'buoy':<6}  " + "  ".join(f"{y:>7}" for y in years) + "   total"
    print(header)
    print("-" * len(header))
    for bid, by_year in coverage.items():
        total = sum(by_year.values())
        cells = "  ".join(f"{by_year.get(y, 0):>7}" for y in years)
        print(f"{bid:<6}  {cells}   {total:>5}")


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.backfill_primary_swell")
    ap.add_argument("--buoy", default=None,
                    help="Limit to one buoy id (default: all 8 CSC buoys).")
    ap.add_argument("--years", nargs="*", type=int, default=None,
                    help="Year list (default: 2024..current).")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Politeness sleep between NDBC requests (s).")
    args = ap.parse_args()
    buoy_ids = [args.buoy] if args.buoy else None
    run_backfill(buoy_ids=buoy_ids, years=args.years, sleep_s=args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
