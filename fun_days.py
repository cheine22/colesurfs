"""
colesurfs — Observed Fun+ day ledger

Backs two dashboard figures:
  • "days since last fun+" (second line of the Fun+ Days cell, per buoy row)
  • "fun+ days this calendar year, by category" (regional-view summary row)

Both are computed from what the buoy actually recorded, using the SAME rule
the forward-looking Fun+ Days column applies to the models
(index.html computeModelOverview): sample every 3 h, skip night, categorize
the primary swell, gate each window on ≥1 region spot with
Textured-or-better wind, and call a day fun+ when ≥2 windows qualify. The
day's single category is the highest tier that ≥2 windows reach, so a
SOLID day is "≥2 windows at SOLID or better" and FUN means fun-but-not-solid.

Inputs (all local, all gitignored):
  .csc_data/observations/buoy=<id>/year=Y/*.parquet   historical NDBC/CDIP obs
  .csc_data/live_log/observations/buoy=<id>/…         30-min live obs (obs_logger)
  .csc_data/wind_archive/year=Y.parquet               per-spot hourly ECMWF wind,
                                                      pulled from Open-Meteo's
                                                      historical-forecast API —
                                                      the same ecmwf_ifs model the
                                                      dashboard's wind cells use

Outputs:
  .csc_data/fun_days/buoy=<id>/year=Y.parquet   one row per local day:
      date, category, fun_windows, obs_windows, day_windows, wind_gated,
      peak_energy (ft²·s, max H²×T over the day's obs — buoy.py's energy
      convention), peak_h_ft, peak_p_s (the reading at that peak),
      p_min / p_max (primary-period span), n_obs

Primary swell is partition=1 (the dashboard's spectral decomposition), with
partition=0 (combined stdmet) as the fallback for hours that have no
spectral row — the same precedence _primaryHP applies to a live record.

Maintenance:
  python fun_days.py --backfill-wind 2025-01-01   one-off wind archive fill
  python fun_days.py --topup --rebuild            daily (com.colesurfs.fun-days)
  python fun_days.py --summary                    print what /api/fun_days serves
  python fun_days.py --rebuild --year 2023        one-off ledger for an older year
                                                  (the /review page also builds
                                                  missing years on demand)
"""
from __future__ import annotations

import statistics
import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone as dtz
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from astral import LocationInfo
from astral.sun import sun

import swell_rules
import wind_rules
from cache import record_api_calls, ttl_cache
from config import SPOTS, WIND_SPOTS, TIMEZONE, WIND_MODELS, ms_to_mph, m_to_ft

ROOT = Path(__file__).resolve().parent
OBS_HIST_DIR = ROOT / ".csc_data" / "observations"
OBS_LIVE_DIR = ROOT / ".csc_data" / "live_log" / "observations"
WIND_ARCHIVE_DIR = ROOT / ".csc_data" / "wind_archive"
LEDGER_DIR = ROOT / ".csc_data" / "fun_days"

HISTORICAL_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WIND_MODEL_ID = WIND_MODELS["EURO"]

TZ = ZoneInfo(TIMEZONE)
STRIDE_H = 3
WINDOW_HOURS = tuple(range(0, 24, STRIDE_H))
OBS_MATCH_TOL_S = 90 * 60          # nearest obs must sit within ±90 min of the window start
MIN_WINDOWS = 2                    # windows needed for a day to earn a tier
LIGHT_PAD = timedelta(minutes=30)  # first/last light = sunrise-30 / sunset+30 (index.html)
SURFABLE_WIND = {"Glassy", "Groomed", "Clean", "Textured"}
LIVE_TAIL_DAYS = 3                 # days re-derived from live obs at request time
PEAK_MIN_H_FT = 0.5                # keeps tiny long-period noise partitions out of the peak
                                   # (0.2 ft @ 27 s outscores 2 ft @ 8 s); readings
                                   # under this height can't be the day's peak

CATS = swell_rules.CATEGORIES
FUN_IDX = CATS.index("FUN")
FT_PER_M = 3.28084


# ─── Wind archive ────────────────────────────────────────────────────────────

def _unique_coords():
    coords, idx_of, spot_to_idx = [], {}, []
    for s in WIND_SPOTS:
        key = (s["lat"], s["lon"])
        if key not in idx_of:
            idx_of[key] = len(coords)
            coords.append(key)
        spot_to_idx.append(idx_of[key])
    return coords, spot_to_idx


def fetch_wind_history(start: date, end: date) -> list[dict]:
    """Hourly ECMWF wind for every WIND_SPOT over [start, end], as flat rows
    {spot, time, speed_mph, direction_deg, gust_mph}. Chunked so no single
    request spans more than ~3 months."""
    coords, spot_to_idx = _unique_coords()
    if not coords:
        return []
    rows: list[dict] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + timedelta(days=91))
        params = {
            "latitude":        ",".join(str(c[0]) for c in coords),
            "longitude":       ",".join(str(c[1]) for c in coords),
            "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "wind_speed_unit": "ms",
            "timezone":        TIMEZONE,
            "start_date":      chunk_start.isoformat(),
            "end_date":        chunk_end.isoformat(),
            "models":          WIND_MODEL_ID,
        }
        record_api_calls("wind_archive", len(coords))
        r = requests.get(HISTORICAL_API, params=params, timeout=60,
                         headers={"User-Agent": "ColeSurfs/1.0"})
        r.raise_for_status()
        raw = r.json()
        if isinstance(raw, dict):
            if raw.get("error"):
                raise RuntimeError(f"open-meteo: {raw.get('reason')}")
            raw = [raw]
        per_point: list[list[dict]] = []
        for pt in raw:
            h = pt.get("hourly", {})
            times = h.get("time", [])
            spd, dirn, gst = (h.get("wind_speed_10m", []), h.get("wind_direction_10m", []),
                              h.get("wind_gusts_10m", []))
            per_point.append([
                {"time": t,
                 "speed_mph": ms_to_mph(spd[j]) if j < len(spd) else None,
                 "direction_deg": dirn[j] if j < len(dirn) else None,
                 "gust_mph": ms_to_mph(gst[j]) if j < len(gst) else None}
                for j, t in enumerate(times)
            ])
        for si, s in enumerate(WIND_SPOTS):
            ui = spot_to_idx[si]
            if ui < len(per_point):
                for rec in per_point[ui]:
                    if rec["speed_mph"] is None:
                        continue
                    rows.append({"spot": s["name"], **rec})
        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end:
            time.sleep(1.0)
    return rows


def _wind_year_path(year: int) -> Path:
    return WIND_ARCHIVE_DIR / f"year={year}.parquet"


def merge_wind_rows(rows: list[dict]) -> dict[int, int]:
    """Upsert rows into the per-year archive, newer fetch winning on
    (spot, time). Returns {year: rows_in_file}."""
    import pandas as pd
    if not rows:
        return {}
    WIND_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["year"] = df["time"].str.slice(0, 4).astype(int)
    out = {}
    for year, part in df.groupby("year"):
        part = part.drop(columns=["year"])
        path = _wind_year_path(int(year))
        if path.exists():
            old = pd.read_parquet(path)
            part = pd.concat([old, part], ignore_index=True)
        part = (part.drop_duplicates(subset=["spot", "time"], keep="last")
                    .sort_values(["spot", "time"]).reset_index(drop=True))
        tmp = path.with_suffix(".tmp")
        part.to_parquet(tmp, index=False, compression="snappy")
        tmp.replace(path)
        out[int(year)] = len(part)
    return out


def backfill_wind(start: date, end: date | None = None) -> dict[int, int]:
    end = end or date.today()
    return merge_wind_rows(fetch_wind_history(start, end))


def topup_wind(days: int = 7) -> dict[int, int]:
    today = date.today()
    return backfill_wind(today - timedelta(days=days), today)


def load_wind(years) -> "pd.DataFrame":
    import pandas as pd
    frames = []
    for y in sorted(set(years)):
        p = _wind_year_path(y)
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=["spot", "time", "speed_mph", "direction_deg", "gust_mph"])
    return pd.concat(frames, ignore_index=True)


def region_wind_hours(region: str, wind_df, live_records: dict | None = None):
    """(surfable_hours, known_hours) for a region: local-time keys
    'YYYY-MM-DDTHH:MM' where ≥1 spot rates Textured-or-better / where any
    spot has a wind record at all. `live_records` ({spot: [rec…]}, the
    dashboard's cached region-wind payload) fills hours the archive lacks."""
    spots = [ws for ws in WIND_SPOTS
             if ws["buoy_region"] == region and ws["shore_normal"] is not None]
    ok, known = set(), set()
    if not spots:
        return ok, known
    seen = set()
    for ws in spots:
        part = wind_df[wind_df["spot"] == ws["name"]] if len(wind_df) else wind_df
        for t, spd, dirn, gst in zip(part["time"], part["speed_mph"],
                                     part["direction_deg"], part["gust_mph"]):
            seen.add((ws["name"], t))
            known.add(t)
            if t in ok:
                continue
            cond = wind_rules.categorize(spd, dirn, ws["shore_normal"], gst)
            if cond in SURFABLE_WIND:
                ok.add(t)
        for rec in (live_records or {}).get(ws["name"], []) or []:
            t = rec.get("time")
            if not t or (ws["name"], t) in seen or rec.get("speed_mph") is None:
                continue
            known.add(t)
            if t in ok:
                continue
            cond = wind_rules.categorize(rec["speed_mph"], rec.get("direction_deg"),
                                         ws["shore_normal"], rec.get("gust_mph"))
            if cond in SURFABLE_WIND:
                ok.add(t)
    return ok, known


# ─── Observations ────────────────────────────────────────────────────────────

_OBS_COLS = {"valid_utc", "partition", "hs_m", "tp_s"}


_MONTH_TAG = re.compile(r"(\d{4})-(\d{2})")


def _obs_files(buoy_id: str, years, months=None) -> list[Path]:
    """Parquet shards for the given years. With `months` ({(y, m), …}) the
    month-tagged shards (spectral-YYYY-MM, stdmet-YYYY-MM, live month=MM
    dirs) outside that set are skipped; untagged shards (yearly, realtime,
    CDIP) are always read."""
    files: list[Path] = []
    for y in sorted(set(years)):
        d = OBS_HIST_DIR / f"buoy={buoy_id}" / f"year={y}"
        if d.is_dir():
            for f in sorted(d.glob("*.parquet")):
                m = _MONTH_TAG.search(f.name)
                if months is not None and m and (int(m.group(1)), int(m.group(2))) not in months:
                    continue
                files.append(f)
        d = OBS_LIVE_DIR / f"buoy={buoy_id}" / f"year={y}"
        if d.is_dir():
            for f in sorted(d.rglob("*.parquet")):
                mdir = next((p for p in f.parts if p.startswith("month=")), None)
                if months is not None and mdir and (y, int(mdir[6:])) not in months:
                    continue
                files.append(f)
    return files


def load_obs(buoy_id: str, years, months=None) -> "pd.DataFrame":
    """Primary-swell series for a buoy: columns t_utc (tz-aware), h_ft, p_s,
    one row per observation timestamp, sorted. partition=1 wins over
    partition=0 at the same timestamp; later ingest wins on duplicates."""
    import pandas as pd
    frames = []
    for f in _obs_files(buoy_id, years, months):
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"[fun_days] unreadable {f.name}: {e}")
            continue
        if not _OBS_COLS.issubset(df.columns):
            continue
        cols = ["valid_utc", "partition", "hs_m", "tp_s"]
        if "ingest_utc" in df.columns:
            cols.append("ingest_utc")
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=["t_utc", "h_ft", "p_s"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["partition"].isin([0, 1])].dropna(subset=["hs_m", "tp_s"])
    if "ingest_utc" in df.columns:
        df = df.sort_values("ingest_utc", na_position="first")
    df = df.drop_duplicates(subset=["valid_utc", "partition"], keep="last")
    # Prefer the spectral primary (partition 1) — stable sort keeps it last.
    df = df.sort_values(["valid_utc", "partition"]).drop_duplicates("valid_utc", keep="last")
    df["t_utc"] = pd.to_datetime(df["valid_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["t_utc"])
    # Match the dashboard's rounding: spectral heights carry 2 dp, stdmet 1 dp.
    df["h_ft"] = [round(float(h) * FT_PER_M, 2) if p == 1 else m_to_ft(float(h))
                  for h, p in zip(df["hs_m"], df["partition"])]
    df["p_s"] = df["tp_s"].astype(float)
    return df[["t_utc", "h_ft", "p_s"]].sort_values("t_utc").reset_index(drop=True)


# ─── Day classification ──────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _light_bounds(lat: float, lon: float, day: date):
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=TIMEZONE)
    try:
        s = sun(loc.observer, date=day, tzinfo=TZ)
    except Exception:
        return None
    return s["sunrise"] - LIGHT_PAD, s["sunset"] + LIGHT_PAD


def _cat_idx(h_ft, p_s) -> int:
    return CATS.index(swell_rules.categorize(h_ft, p_s))


def classify_day(day: date, lat: float, lon: float, obs_epoch, obs_h, obs_p,
                 wind_ok: set, wind_known: set) -> dict:
    """One ledger row for a local calendar day. `obs_epoch` is a sorted
    numpy array of UTC epoch seconds aligned with obs_h / obs_p."""
    import numpy as np
    bounds = _light_bounds(lat, lon, day)
    day_windows = obs_windows = 0
    wind_gated = False
    tier_hits = [0] * len(CATS)          # windows at tier ≥ i that pass the gate
    for hh in WINDOW_HOURS:
        t_local = datetime(day.year, day.month, day.day, hh, tzinfo=TZ)
        t_end = t_local + timedelta(hours=STRIDE_H)
        if bounds and (t_end <= bounds[0] or t_local >= bounds[1]):
            continue
        day_windows += 1
        if not len(obs_epoch):
            continue
        ep = t_local.timestamp()
        i = int(np.searchsorted(obs_epoch, ep))
        cands = [j for j in (i - 1, i) if 0 <= j < len(obs_epoch)]
        j = min(cands, key=lambda k: abs(obs_epoch[k] - ep)) if cands else None
        if j is None or abs(obs_epoch[j] - ep) > OBS_MATCH_TOL_S:
            continue
        obs_windows += 1
        ci = _cat_idx(float(obs_h[j]), float(obs_p[j]))
        key = t_local.strftime("%Y-%m-%dT%H:%M")
        if key in wind_known:
            wind_gated = True
            if key not in wind_ok:
                continue
        for k in range(ci + 1):
            tier_hits[k] += 1
    if obs_windows == 0:
        category = None
    else:
        best = 0
        for k in range(len(CATS)):
            if tier_hits[k] >= MIN_WINDOWS:
                best = k
        category = CATS[best]
    # Daily peak over EVERY obs in the local day (not just the 3 h windows):
    # the /review energy and period charts plot the day's strongest reading.
    peak = {"peak_energy": None, "peak_h_ft": None, "peak_p_s": None,
            "p_min": None, "p_max": None, "n_obs": 0}
    if len(obs_epoch):
        d0 = datetime(day.year, day.month, day.day, tzinfo=TZ).timestamp()
        d1 = (datetime(day.year, day.month, day.day, tzinfo=TZ) + timedelta(days=1)).timestamp()
        a, b = int(np.searchsorted(obs_epoch, d0)), int(np.searchsorted(obs_epoch, d1))
        if b > a:
            hh, pp = obs_h[a:b].astype(float), obs_p[a:b].astype(float)
            e = np.where(hh >= PEAK_MIN_H_FT, hh * hh * pp, -1.0)   # H²·T, as buoy.py
            k = int(np.argmax(e))
            real = hh >= PEAK_MIN_H_FT
            peak = {"peak_energy": round(float(e[k]), 1) if e[k] >= 0 else None,
                    "peak_h_ft": float(hh[k]) if e[k] >= 0 else None,
                    "peak_p_s": float(pp[k]) if e[k] >= 0 else None,
                    "p_min": float(pp[real].min()) if real.any() else None,
                    "p_max": float(pp[real].max()) if real.any() else None,
                    "n_obs": int(b - a)}
    return {
        "date":        day.isoformat(),
        "category":    category,
        "fun_windows": tier_hits[FUN_IDX],
        "obs_windows": obs_windows,
        "day_windows": day_windows,
        "wind_gated":  wind_gated,
        **peak,
    }


def _buoy_meta(buoy_id: str) -> dict:
    for s in SPOTS:
        if s["buoy_id"] == buoy_id:
            return s
    raise KeyError(f"buoy {buoy_id!r} is not in regions.yaml")


def classify_range(buoy_id: str, start: date, end: date, *, obs=None,
                   wind_df=None, live_wind: dict | None = None) -> list[dict]:
    import numpy as np
    meta = _buoy_meta(buoy_id)
    years = {start.year, end.year}
    if obs is None:
        months = set()
        d = date(start.year, start.month, 1)
        while d <= end:
            months.add((d.year, d.month))
            d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        obs = load_obs(buoy_id, years, months)
    if wind_df is None:
        wind_df = load_wind(years)
    ok, known = region_wind_hours(meta["name"], wind_df, live_wind)
    import pandas as pd
    # total_seconds is unit-agnostic (parquet round-trips can land in ms, not ns).
    epoch = ((obs["t_utc"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
             if len(obs) else np.array([]))
    h = obs["h_ft"].to_numpy() if len(obs) else np.array([])
    p = obs["p_s"].to_numpy() if len(obs) else np.array([])
    rows = []
    d = start
    while d <= end:
        rows.append(classify_day(d, meta["lat"], meta["lon"], epoch, h, p, ok, known))
        d += timedelta(days=1)
    return rows


# ─── Ledger ──────────────────────────────────────────────────────────────────

def _ledger_path(buoy_id: str, year: int) -> Path:
    return LEDGER_DIR / f"buoy={buoy_id}" / f"year={year}.parquet"


def rebuild_ledger(buoy_id: str, year: int, *, wind_df=None) -> int:
    import pandas as pd
    today = datetime.now(TZ).date()
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), today)
    if start > end:
        return 0
    rows = classify_range(buoy_id, start, end, wind_df=wind_df)
    path = _ledger_path(buoy_id, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(path)
    return len(df)


def load_ledger(buoy_id: str, year: int) -> list[dict]:
    import pandas as pd
    p = _ledger_path(buoy_id, year)
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict("records")


def rebuild_all(years=None) -> dict:
    today = datetime.now(TZ).date()
    years = sorted(years or {today.year, today.year - 1})
    wind_df = load_wind(years)
    out = {}
    for s in SPOTS:
        for y in years:
            try:
                out[f"{s['buoy_id']}/{y}"] = rebuild_ledger(s["buoy_id"], y, wind_df=wind_df)
            except Exception as e:
                out[f"{s['buoy_id']}/{y}"] = f"error: {e}"
    return out


# ─── Summary (what /api/fun_days serves) ─────────────────────────────────────

def rows_between(buoy_id: str, start: date, end: date, *,
                 live_wind: dict | None = None, today: date | None = None) -> dict[str, dict]:
    """{date_iso: ledger_row} over [start, end]. Missing ledger years are
    built on demand (skipped when the buoy has no obs shards for that year);
    the trailing LIVE_TAIL_DAYS are re-derived from live obs so "today" and
    "yesterday" reflect readings the nightly rebuild hasn't seen yet."""
    today = today or datetime.now(TZ).date()
    end = min(end, today)
    by_date: dict[str, dict] = {}
    for y in range(start.year, end.year + 1):
        if not _ledger_path(buoy_id, y).exists():
            if not _obs_files(buoy_id, {y}):
                continue
            try:
                rebuild_ledger(buoy_id, y)
            except Exception as e:
                print(f"[fun_days] ledger build {buoy_id}/{y} failed: {e}")
        for r in load_ledger(buoy_id, y):
            if start.isoformat() <= r["date"] <= end.isoformat():
                by_date[r["date"]] = r
    tail_start = max(start, today - timedelta(days=LIVE_TAIL_DAYS - 1))
    if tail_start <= end:
        try:
            for r in classify_range(buoy_id, tail_start, end, live_wind=live_wind):
                by_date[r["date"]] = r
        except Exception as e:
            print(f"[fun_days] live tail {buoy_id} failed: {e}")
    return by_date


def _fun_runs(fun_dates) -> list[tuple[date, date]]:
    """Maximal runs of consecutive fun+ calendar days — one run is one
    fun+ swell, however many days it lasted."""
    runs: list[tuple[date, date]] = []
    for d in sorted(fun_dates):
        if runs and d == runs[-1][1] + timedelta(days=1):
            runs[-1] = (runs[-1][0], d)
        else:
            runs.append((d, d))
    return runs


def droughts(fun_dates) -> list[dict]:
    """Gaps between consecutive fun+ swells: the non-fun+ days from the day
    after one run ends to the day before the next begins. Days without obs
    inside a gap count as drought days (calendar span). review.html's
    `droughtsOf` mirrors this rule client-side."""
    runs = _fun_runs(fun_dates)
    return [{"days": (b[0] - a[1]).days - 1,
             "start": (a[1] + timedelta(days=1)).isoformat(),
             "end": (b[0] - timedelta(days=1)).isoformat()}
            for a, b in zip(runs, runs[1:])]


def summary(buoy_id: str, *, live_wind: dict | None = None, today: date | None = None) -> dict:
    """Days since the last fun+ day and this year's tally — from the ledger
    for this year and last, live tail included (see rows_between)."""
    today = today or datetime.now(TZ).date()
    by_date = rows_between(buoy_id, date(today.year - 1, 1, 1), today,
                           live_wind=live_wind, today=today)

    def _idx(r):
        return CATS.index(r["category"]) if r and r.get("category") else -1

    last_fun = None
    d = today
    earliest = None
    while d.year >= today.year - 1:
        r = by_date.get(d.isoformat())
        if r and r.get("category"):
            earliest = d
            if _idx(r) >= FUN_IDX:
                last_fun = r
                break
        d -= timedelta(days=1)

    year_rows = [r for k, r in by_date.items() if k.startswith(f"{today.year}-")]
    with_data = [r for r in year_rows if r.get("category")]
    by_cat = {c: 0 for c in CATS[FUN_IDX:]}
    for r in with_data:
        if _idx(r) >= FUN_IDX:
            by_cat[r["category"]] += 1
    today_row = by_date.get(today.isoformat())

    # Longest drought in the trailing year. The open-ended gap since the last
    # fun+ day competes with the closed ones; a 0-day "current drought" (today
    # is fun+) never wins, and no line at all when the year holds no fun+ day.
    win_start = today - timedelta(days=364)
    fun_dates = {date.fromisoformat(k) for k, r in by_date.items()
                 if k >= win_start.isoformat() and _idx(r) >= FUN_IDX}
    gaps = droughts(fun_dates)
    longest = max(gaps, key=lambda g: g["days"], default=None)
    drought = None
    if last_fun and date.fromisoformat(last_fun["date"]) >= win_start:
        current = (today - date.fromisoformat(last_fun["date"])).days
        if current > 0 and current >= (longest["days"] if longest else 0):
            drought = {"current_is_longest": True, "days": current,
                       "start": (date.fromisoformat(last_fun["date"]) + timedelta(days=1)).isoformat(),
                       "end": today.isoformat()}
        elif longest:
            drought = {"current_is_longest": False, **longest}
    return {
        "buoy_id":       buoy_id,
        "today":         today.isoformat(),
        "today_cat":     today_row.get("category") if today_row else None,
        "days_since_fun": (today - date.fromisoformat(last_fun["date"])).days if last_fun else None,
        "last_fun_date": last_fun["date"] if last_fun else None,
        "last_fun_cat":  last_fun["category"] if last_fun else None,
        # Earliest day with obs in the walk — when days_since_fun is null this
        # bounds the claim ("no fun+ day since at least …").
        "searched_from": earliest.isoformat() if earliest else None,
        "drought":       drought,
        "year": {
            "year":           today.year,
            "fun_plus":       sum(by_cat.values()),
            "by_cat":         by_cat,
            "days_with_data": len(with_data),
            "days_elapsed":   today.timetuple().tm_yday,
            "first_data_date": min((r["date"] for r in with_data), default=None),
        },
    }


@ttl_cache(ttl_seconds=600, skip_none=True)
def all_summaries() -> dict | None:
    """{buoy_id: summary} for every dashboard buoy. Uses the dashboard's
    cached EURO region-wind payload to gate today's hours."""
    live_wind = _live_wind()
    out = {}
    for s in SPOTS:
        try:
            out[s["buoy_id"]] = summary(s["buoy_id"], live_wind=live_wind)
        except Exception as e:
            print(f"[fun_days] summary {s['buoy_id']} failed: {type(e).__name__}: {e}")
            out[s["buoy_id"]] = None
    return out or None


def _live_wind():
    try:
        from wind import fetch_region_wind_forecasts
        return fetch_region_wind_forecasts("EURO") or None
    except Exception as e:
        print(f"[fun_days] live wind unavailable: {e}")
        return None


@ttl_cache(ttl_seconds=600, skip_none=True)
def review_payload(start_iso: str, end_iso: str) -> dict | None:
    """/api/review: {buoy_id: [ledger rows]} for every dashboard buoy over
    [start, end] (end clamped to today), rows in date order. Buoys with no
    obs in the window get an empty list — the page says so rather than
    drawing an empty tally as zeros."""
    start, end = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
    if end < start:
        return None
    live_wind = _live_wind()
    out = {}
    for s in SPOTS:
        try:
            rows = rows_between(s["buoy_id"], start, end, live_wind=live_wind)
            out[s["buoy_id"]] = [rows[k] for k in sorted(rows)]
        except Exception as e:
            print(f"[fun_days] review {s['buoy_id']} failed: {type(e).__name__}: {e}")
            out[s["buoy_id"]] = []
    return {"start": start.isoformat(), "end": min(end, datetime.now(TZ).date()).isoformat(),
            "buoys": out}


# ─── Season history (/review bottom panel) ───────────────────────────────────

# Seasons run equinox to solstice on fixed dates (not calendar months):
# winter Dec 21 → Mar 20, spring Mar 21 → Jun 20, summer Jun 21 → Sep 20,
# fall Sep 21 → Dec 20. Winter is named for the year of its Jan–Mar part.
# review.html's seasonRange() must agree.
SEASON_START = {"winter": (12, 21), "spring": (3, 21), "summer": (6, 21), "fall": (9, 21)}
_NEXT_SEASON = {"winter": "spring", "spring": "summer", "summer": "fall", "fall": "winter"}
SEASONS_FIRST_YEAR = 2019   # NDBC yearly backfills + wind archive reach here;
                            # the first season shown is FALL 2019 (winter 2019
                            # would need Dec 2018, which isn't archived)


def season_of(d: date) -> tuple[str, int]:
    md = (d.month, d.day)
    if md >= SEASON_START["winter"]:
        return "winter", d.year + 1
    for s in ("fall", "summer", "spring"):
        if md >= SEASON_START[s]:
            return s, d.year
    return "winter", d.year


def _season_bounds(season: str, year: int):
    m, d = SEASON_START[season]
    start = date(year - 1 if season == "winter" else year, m, d)
    nm, nd = SEASON_START[_NEXT_SEASON[season]]
    end = date(year, nm, nd) - timedelta(days=1)
    return start, end


def _ledger_years(buoy_id: str) -> list[int]:
    d = LEDGER_DIR / f"buoy={buoy_id}"
    if not d.is_dir():
        return []
    return sorted(int(p.stem.split("=")[1]) for p in d.glob("year=*.parquet"))


@ttl_cache(ttl_seconds=3600, skip_none=True)
def season_tables() -> dict | None:
    """Per buoy, per season, one row per year with fun+ / flat / solid /
    firing day counts (fun+ = FUN or better; flat, solid, firing = exactly
    that tier) and the median drought (days between fun+ swells, see
    `droughts`) within the season,
    from every ledger year on disk, fall SEASONS_FIRST_YEAR onward. `elapsed`
    is the season's day count to date so the page can flag partial coverage."""
    today = datetime.now(TZ).date()
    out = {}
    for s in SPOTS:
        bid = s["buoy_id"]
        per: dict[tuple, dict] = {}
        for y in _ledger_years(bid):
            if y < SEASONS_FIRST_YEAR - 1:
                continue
            for r in load_ledger(bid, y):
                cat = r.get("category")
                if not cat:
                    continue
                season, syear = season_of(date.fromisoformat(r["date"]))
                if syear < SEASONS_FIRST_YEAR or (syear == SEASONS_FIRST_YEAR and season != "fall"):
                    continue
                c = per.setdefault((season, syear),
                                   {"year": syear, "days": 0, "fun_plus": 0, "flat": 0,
                                    "solid": 0, "firing": 0, "_fun_dates": set()})
                c["days"] += 1
                c["fun_plus"] += CATS.index(cat) >= FUN_IDX
                if CATS.index(cat) >= FUN_IDX:
                    c["_fun_dates"].add(date.fromisoformat(r["date"]))
                c["flat"] += cat == "FLAT"
                c["solid"] += cat == "SOLID"
                c["firing"] += cat == "FIRING"
        tables = {}
        for season in ("fall", "winter", "spring", "summer"):
            rows = []
            for (sn, syear), c in per.items():
                if sn != season:
                    continue
                start, end = _season_bounds(season, syear)
                gaps = [g["days"] for g in droughts(c.pop("_fun_dates"))]
                rows.append({**c, "season_days": (end - start).days + 1,
                             "elapsed": max(0, (min(end, today) - start).days + 1),
                             "median_drought": statistics.median(gaps) if gaps else None})
            tables[season] = sorted(rows, key=lambda r: -r["year"])
        out[bid] = tables
    return {"first_year": SEASONS_FIRST_YEAR, "buoys": out}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(prog="fun_days")
    ap.add_argument("--backfill-wind", metavar="YYYY-MM-DD",
                    help="Fill the wind archive from this date through today.")
    ap.add_argument("--topup", action="store_true",
                    help="Refresh the last 7 days of the wind archive.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Recompute every buoy's ledger for this year and last.")
    ap.add_argument("--year", type=int, default=None,
                    help="With --rebuild: a single year instead of this+last.")
    ap.add_argument("--summary", action="store_true",
                    help="Print the per-buoy summary payload.")
    args = ap.parse_args()
    t0 = time.monotonic()
    stamp = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.backfill_wind:
        n = backfill_wind(date.fromisoformat(args.backfill_wind))
        print(f"[fun_days] wind archive rows/year: {n}")
    if args.topup:
        n = topup_wind()
        print(f"[fun_days] wind top-up rows/year: {n}")
    if args.rebuild:
        res = rebuild_all([args.year] if args.year else None)
        for k, v in res.items():
            print(f"  {k}: {v}")
    if args.summary:
        import json
        for s in SPOTS:
            print(s["name"], json.dumps(summary(s["buoy_id"]), indent=None))
    if not any([args.backfill_wind, args.topup, args.rebuild, args.summary]):
        ap.print_help()
        return 1
    (ROOT / ".csc2_data" / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / ".csc2_data" / "logs" / "fun_days.log").open("a") as f:
        f.write(f"[{stamp}] {' '.join(sys.argv[1:])} elapsed={time.monotonic()-t0:.1f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
