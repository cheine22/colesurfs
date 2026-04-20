"""NDBC historical stdmet parser — emits one dict per data row.

Historical stdmet archives live at
  https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz

Row format (header + units + rows):
  #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES ATMP WTMP DEWP VIS TIDE
  #yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT  ...
  2025 01 01 00 00 169  4.7  5.9 99.00 99.00 99.00 999   ...

Missing values: 99.00 for numerics, 999 for directions, 9999 for pressure.
Our emitter keeps only rows with valid WVHT + DPD; each row becomes a dict
of {valid_utc, wvht_m, dpd_s, mwd_deg, apd_s, wspd_ms, wdir_deg, pres_hpa}.
"""

from __future__ import annotations

import gzip
import io
from datetime import datetime, timezone
from typing import Iterator

import requests

NDBC_STDMET_HIST_URL = (
    "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"
)


def _safe(v: str) -> float | None:
    if v in ("MM", "m", None, ""):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    # Convention-specific sentinels for NDBC
    if v in ("99.00", "99.0", "99", "999.0", "999", "9999.0", "9999"):
        # Could be a real value (e.g. pressure 999.X doesn't exist) — be
        # conservative and treat the tokens above as missing.
        return None
    return x


def _safe_dir(v: str) -> float | None:
    if v in ("MM", "m", None, "", "999"):
        return None
    try:
        deg = float(v)
    except ValueError:
        return None
    if deg >= 990:       # 999 sentinel for missing direction
        return None
    return deg


def iter_stdmet_rows(text: str) -> Iterator[dict]:
    """Iterate every valid row in a decompressed stdmet archive file."""
    lines = text.splitlines()
    headers = None
    for line in lines:
        if line.startswith("#"):
            if headers is None:
                headers = line.lstrip("# ").split()
            continue
        if not line.strip():
            continue
        if headers is None:
            continue
        parts = line.split()
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, parts))
        try:
            ts = datetime(
                int(row["YY"]), int(row["MM"]), int(row["DD"]),
                int(row["hh"]), int(row["mm"]),
                tzinfo=timezone.utc,
            )
        except (KeyError, ValueError):
            continue

        wvht = _safe(row.get("WVHT", ""))
        dpd = _safe(row.get("DPD", ""))
        if wvht is None:      # no wave height → not useful as CSC target
            continue

        yield {
            "valid_utc": ts,
            "wvht_m": wvht,
            "dpd_s": dpd,
            "apd_s": _safe(row.get("APD", "")),
            "mwd_deg": _safe_dir(row.get("MWD", "")),
            "wspd_ms": _safe(row.get("WSPD", "")),
            "wdir_deg": _safe_dir(row.get("WDIR", "")),
            "pres_hpa": _safe(row.get("PRES", "")),
            "wtmp_c": _safe(row.get("WTMP", "")),
        }


def fetch_stdmet_year(station: str, year: int, timeout_s: float = 60.0
                      ) -> list[dict]:
    """Download + gunzip + parse one (station, year) archive file."""
    url = NDBC_STDMET_HIST_URL.format(station=station, year=year)
    r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "ColeSurfs/1.0"})
    if r.status_code == 404:
        return []
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    return list(iter_stdmet_rows(text))
