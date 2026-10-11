"""colesurfs — gridded 10 m wind for the map (v2.0).

The map's wind field comes straight from the models' own 0.25° GRIB output,
not from Open-Meteo point queries (a dense grid would burn the free quota in
one call): GFS through NOMADS' grib filter (a subregion cut of UGRD/VGRD,
~23 KB a step; AWS `noaa-gfs-bdp-pds` byte ranges as the fallback) and ECMWF
IFS through the open-data index files + HTTP byte ranges (one global 10u /
10v field per step, ~0.75 MB each; data.ecmwf.int, AWS mirror as fallback).
Each step carries 10u / 10v and the model's gust (GFS `GUST` at the
surface, ECMWF `10fg`), decoded with eccodes, cut to ENVELOPE, quantised to
0.05 m/s int16 and kept under .cache/wind_field/<MODEL>/<YYYYMMDDHH>.npz; a
run is written as soon as its first steps land and topped up while the
source is still publishing. `series()` composites the newest run for every
valid hour — older runs fill the past hours and, behind a short 06/18Z ECMWF
run, the tail — and the dashboard fetches it as gzip'd int16 chunks.
`sample()` reads the same series at points, with the page's own
interpolation (linear between steps, bilinear in the grid), so the table's
spot winds are the field the map paints.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests

from cache import _DISK_CACHE_DIR, record_api_calls
from config import TIMEZONE

# ── grid: 0.25°, rows north → south, columns west → east ─────────────────────
LAT_N, LAT_S, LON_W, LON_E = 48.0, 30.0, -82.0, -55.0
STEP = 0.25
NY = int(round((LAT_N - LAT_S) / STEP)) + 1          # 73
NX = int(round((LON_E - LON_W) / STEP)) + 1          # 109
GRID = {"la1": LAT_N, "lo1": LON_W, "nx": NX, "ny": NY, "dx": STEP, "dy": STEP}
SCALE = 20                                           # int16 units of 0.05 m/s

DIR = os.path.join(_DISK_CACHE_DIR, "wind_field")
KEEP_DAYS = 4                                        # the composite only reaches back 3 days
UA = {"User-Agent": "ColeSurfs/2.0 (surfreport.coleheine.com)"}

# GFS: hourly to f120, 3-hourly to f240 (0.25° pgrb2 is hourly only to 120).
GFS_STEPS = list(range(0, 121)) + list(range(123, 241, 3))
# ECMWF IFS open data: 3-hourly to 144, 6-hourly to 240 (00/12Z); 06/18Z stop at 90.
ECMWF_STEPS_LONG = list(range(0, 145, 3)) + list(range(150, 241, 6))
ECMWF_STEPS_SHORT = list(range(0, 91, 3))

NOMADS = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
GFS_AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
# ECMWF's own server throttles (429) and asks regular users to prefer the
# cloud mirrors; the AWS mirror answers bursts with 503 Slow Down. Google's
# mirror first, AWS next, data.ecmwf.int last.
ECMWF_ROOTS = ("https://storage.googleapis.com/ecmwf-open-data",
               "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
               "https://data.ecmwf.int/forecasts")

# A cycle is probed this long after its init (first 0.25° GFS files land
# ~3 h 20 min after init, ECMWF open data ~6 h 30 min).
READY_AFTER = {"GFS": timedelta(hours=3, minutes=30), "EURO": timedelta(hours=6, minutes=40)}
CYCLE_HOURS = {"GFS": (0, 6, 12, 18), "EURO": (0, 6, 12, 18)}

_lock = threading.RLock()
_runs: dict[str, dict[str, dict]] = {"GFS": {}, "EURO": {}}   # model → run id → {init, steps, u, v}
_series_cache: dict[str, dict] = {}
_status: dict[str, dict] = {"GFS": {}, "EURO": {}}


# ── GRIB decoding ─────────────────────────────────────────────────────────────

def _split_grib2(buf: bytes):
    """Yield the GRIB2 messages in a byte string (section 0 carries the length)."""
    pos = 0
    while True:
        i = buf.find(b"GRIB", pos)
        if i < 0 or i + 16 > len(buf):
            return
        n = int.from_bytes(buf[i + 8:i + 16], "big")
        if n <= 0 or i + n > len(buf):
            return
        yield buf[i:i + n]
        pos = i + n


def _decode(msg: bytes) -> dict:
    """One GRIB2 message → {name, lat (asc/desc as stored), lon, values[nj, ni]}."""
    import eccodes as ec
    gid = ec.codes_new_from_message(msg)
    try:
        name = ec.codes_get(gid, "shortName")
        ni, nj = ec.codes_get(gid, "Ni"), ec.codes_get(gid, "Nj")
        lats = ec.codes_get_array(gid, "distinctLatitudes")
        lons = ec.codes_get_array(gid, "distinctLongitudes")
        vals = ec.codes_get_values(gid).reshape(nj, ni)
        if ec.codes_get(gid, "jScansPositively"):
            vals = vals[::-1]                          # rows north → south
            lats = lats[::-1] if lats[0] < lats[-1] else lats
        else:
            lats = lats[::-1] if lats[0] < lats[-1] else lats
        lons = np.where(lons > 180, lons - 360, lons)
        return {"name": name, "lats": np.asarray(lats, float), "lons": np.asarray(lons, float),
                "values": np.asarray(vals, np.float32)}
    finally:
        ec.codes_release(gid)


def _cut(field: dict) -> np.ndarray:
    """Cut a decoded field to the envelope grid (rows N→S, cols W→E)."""
    lats, lons, v = field["lats"], field["lons"], field["values"]
    if lats[0] < lats[-1]:
        lats, v = lats[::-1], v[::-1]
    order = np.argsort(lons)
    lons, v = lons[order], v[:, order]
    r0 = int(np.argmin(np.abs(lats - LAT_N)))
    c0 = int(np.argmin(np.abs(lons - LON_W)))
    out = v[r0:r0 + NY, c0:c0 + NX]
    if out.shape != (NY, NX) or abs(lats[r0] - LAT_N) > 1e-3 or abs(lons[c0] - LON_W) > 1e-3:
        raise ValueError(f"grid mismatch: got {out.shape} at lat {lats[r0]} lon {lons[c0]}")
    return out


def _uvg_from(buf: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(u, v, gust) from the GRIB messages in `buf`; gust None when absent."""
    u = v = g = None
    for msg in _split_grib2(buf):
        f = _decode(msg)
        if f["name"] in ("10u", "u"):
            u = _cut(f)
        elif f["name"] in ("10v", "v"):
            v = _cut(f)
        elif f["name"] == "gust" or f["name"].startswith(("10fg", "fg")):
            g = _cut(f)                                # GFS `gust`; ECMWF 10fg / 10fg3 / 10fg6
    if u is None or v is None:
        raise ValueError("u/v missing from GRIB")
    return u, v, g


def _q(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a * SCALE), -32767, 32767).astype(np.int16)


NO_GUST = np.int16(-1)                               # gust unknown for this step


def _q_gust(g: np.ndarray | None) -> np.ndarray:
    """Gust field → int16, or NO_GUST everywhere when the step has none (an
    ECMWF +0 h `10fg` is a 224-byte constant field — nothing to show)."""
    if g is None or not np.isfinite(g).any() or float(np.nanmax(g)) <= 0:
        return np.full((NY, NX), NO_GUST, np.int16)
    return np.clip(np.round(np.nan_to_num(g, nan=0.0) * SCALE), 0, 32767).astype(np.int16)


# ── fetchers: one (init, step) → (u, v, gust | None) float32 [NY, NX] ─────────

def _get(url: str, label: str, timeout=60, headers=None, ok404=True) -> bytes | None:
    h = dict(UA)
    if headers:
        h.update(headers)
    record_api_calls(label, 1)
    r = requests.get(url, headers=h, timeout=timeout)
    if r.status_code == 404 and ok404:
        return None
    r.raise_for_status()
    return r.content


def _gfs_nomads(init: datetime, step: int) -> tuple | None:
    url = (f"{NOMADS}?dir=%2Fgfs.{init:%Y%m%d}%2F{init:%H}%2Fatmos"
           f"&file=gfs.t{init:%H}z.pgrb2.0p25.f{step:03d}"
           f"&var_UGRD=on&var_VGRD=on&var_GUST=on&lev_10_m_above_ground=on&lev_surface=on"
           f"&subregion=&toplat={LAT_N:g}&leftlon={LON_W % 360:g}&rightlon={LON_E % 360:g}&bottomlat={LAT_S:g}")
    buf = _get(url, "wind_field_gfs", timeout=60)
    if not buf or not buf.startswith(b"GRIB"):
        return None                                    # not published yet (filter returns HTML)
    return _uvg_from(buf)




def _gfs_aws(init: datetime, step: int) -> tuple | None:
    base = f"{GFS_AWS}/gfs.{init:%Y%m%d}/{init:%H}/atmos/gfs.t{init:%H}z.pgrb2.0p25.f{step:03d}"
    idx = _get(base + ".idx", "wind_field_gfs_aws", timeout=30)
    if not idx:
        return None
    lines = idx.decode("ascii", "replace").splitlines()
    starts = [int(l.split(":")[1]) for l in lines]
    ranges = []
    for i, l in enumerate(lines):
        if ":UGRD:10 m above ground:" in l or ":VGRD:10 m above ground:" in l or ":GUST:surface:" in l:
            a = starts[i]
            b = starts[i + 1] - 1 if i + 1 < len(starts) else ""
            ranges.append(f"{a}-{b}")
    if len(ranges) < 2:
        return None
    buf = b"".join(_get(base, "wind_field_gfs_aws", timeout=120, headers={"Range": f"bytes={r}"}, ok404=False)
                   for r in ranges)
    return _uvg_from(buf)


def _gfs_step(init, step):
    try:
        out = _gfs_nomads(init, step)
        if out is not None:
            return out
    except Exception as e:
        print(f"[wind_field] GFS {init:%Y%m%d%H} f{step:03d} NOMADS: {e}", flush=True)
    return _gfs_aws(init, step)


def _ecmwf_step(init: datetime, step: int) -> tuple | None:
    stem = f"{init:%Y%m%d}/{init:%H}z/ifs/0p25/oper/{init:%Y%m%d%H}0000-{step}h-oper-fc"
    last = None
    for root in ECMWF_ROOTS:
        try:
            idx = _get(f"{root}/{stem}.index", "wind_field_ecmwf", timeout=30)
            if idx is None:
                last = None
                continue                               # not on this mirror (yet) — try the next
            recs = [json.loads(l) for l in idx.decode().splitlines() if l.strip()]
            sfc = [r for r in recs if r.get("levtype") == "sfc"]
            want = [r for r in sfc if r.get("param") in ("10u", "10v")]
            if len(want) != 2:
                raise ValueError("10u/10v not in index")
            # the gust is `10fg` to +90 h and at the 6-hourly tail, `10fg3`
            # (3 h max) from +93 to +144 — take whichever this step has
            gusts = [r for r in sfc if str(r.get("param", "")).startswith("10fg")]
            if gusts:
                want.append(min(gusts, key=lambda r: len(r["param"])))
            buf = b"".join(_get(f"{root}/{stem}.grib2", "wind_field_ecmwf", timeout=120,
                                headers={"Range": f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"},
                                ok404=False)
                           for r in want)
            return _uvg_from(buf)
        except Exception as e:
            last = e
            print(f"[wind_field] EURO {init:%Y%m%d%H} +{step}h {root.split('/')[2]}: {e}", flush=True)
    if last is None:
        return None                                    # not published on any mirror yet
    raise RuntimeError(f"ECMWF step failed on every mirror: {last}")


FETCH = {"GFS": _gfs_step, "EURO": _ecmwf_step}


def _planned_steps(model: str, init: datetime) -> list[int]:
    if model == "GFS":
        return GFS_STEPS
    return ECMWF_STEPS_LONG if init.hour in (0, 12) else ECMWF_STEPS_SHORT


# ── run store ─────────────────────────────────────────────────────────────────

def _run_id(init: datetime) -> str:
    return f"{init:%Y%m%d%H}"


def _path(model: str, rid: str) -> str:
    return os.path.join(DIR, model, f"{rid}.npz")


def _save(model: str, run: dict):
    p = _path(model, run["id"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, steps=np.asarray(run["steps"], np.int16), u=run["u"], v=run["v"], g=run["g"],
                 gok=np.asarray(run["gok"], bool),
                 init=np.asarray(run["init"].timestamp()), complete=np.asarray(run["complete"]))
    os.replace(tmp, p)


def _load_disk(model: str):
    d = os.path.join(DIR, model)
    if not os.path.isdir(d):
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npz"):
            continue
        p = os.path.join(d, fn)
        try:
            with np.load(p) as z:
                init = datetime.fromtimestamp(float(z["init"]), timezone.utc)
                if init < cutoff:
                    os.remove(p)
                    continue
                steps = [int(s) for s in z["steps"]]
                # Files written before gusts (no `g` / `gok`): every step reads
                # as not-yet-fetched, so the updater refills the newest cycles
                # with gusts and the older ones serve the past gust-less.
                g = z["g"] if "g" in z.files else np.full((len(steps), NY, NX), NO_GUST, np.int16)
                gok = [bool(x) for x in z["gok"]] if "gok" in z.files else [False] * len(steps)
                # a step past +0 h fetched without a gust (the fetcher once
                # missed ECMWF's `10fg3` naming) is fetched again
                gok = [ok and not (st > 0 and bool((g[i] < 0).all())) for i, (st, ok) in enumerate(zip(steps, gok))]
                run = {"id": fn[:-4], "init": init, "steps": steps,
                       "u": z["u"], "v": z["v"], "g": g, "gok": gok,
                       "complete": bool(z["complete"]) and all(gok)}
            with _lock:
                _runs[model][run["id"]] = run
        except Exception as e:
            print(f"[wind_field] dropping unreadable {p}: {e}", flush=True)
            try:
                os.remove(p)
            except OSError:
                pass
    with _lock:
        _series_cache.pop(model, None)


def _prune(model: str):
    cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
    for rid, run in list(_runs[model].items()):
        if run["init"] < cutoff:
            del _runs[model][rid]
            try:
                os.remove(_path(model, rid))
            except OSError:
                pass


def _candidates(model: str, now: datetime) -> list[datetime]:
    """Cycle inits that should be published by now, newest first (two days back)."""
    out = []
    for d in range(0, 3):
        day = (now - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        for h in CYCLE_HOURS[model]:
            init = day.replace(hour=h)
            if init + READY_AFTER[model] <= now:
                out.append(init)
    return sorted(out, reverse=True)


def _fetch_run(model: str, init: datetime, pause: float) -> bool:
    """Fetch (or top up) one run. Returns True if any new step landed.
    Stops at the first unpublished step past +24 h — sources publish in
    order, so the rest isn't there yet; the next pass resumes."""
    rid = _run_id(init)
    with _lock:
        run = _runs[model].get(rid)
    planned = _planned_steps(model, init)
    # a step counts as fetched only once this fetcher (with gusts) has read it
    have = {s for s, ok in zip(run["steps"], run["gok"]) if ok} if run else set()
    todo = [s for s in planned if s not in have]
    if not todo:
        return False
    got = {}
    fetch = FETCH[model]
    t0 = time.monotonic()
    misses = 0
    for s in todo:
        try:
            uvg = fetch(init, s)
        except Exception as e:
            print(f"[wind_field] {model} {rid} +{s}h: {e}", flush=True)
            if s <= 6 and not got and not run:
                return False                           # source down — try again later
            break                                      # sources publish in order; resume next pass
        if uvg is None:
            if not got and run is None:
                return False                           # run not started
            misses += 1
            if misses >= 3 or s <= 24:
                break                                  # not published yet (a lone skipped file is tolerated)
            continue
        misses = 0
        got[s] = uvg
        if pause:
            time.sleep(pause)
    if not got:
        return False
    with _lock:
        run = _runs[model].get(rid)
        old = set(run["steps"]) if run else set()
        steps = sorted(old | set(got))
        ny, nx = NY, NX
        u = np.zeros((len(steps), ny, nx), np.int16)
        v = np.zeros((len(steps), ny, nx), np.int16)
        g = np.full((len(steps), ny, nx), NO_GUST, np.int16)
        gok = []
        for i, s in enumerate(steps):
            if s in got:
                u[i], v[i], g[i] = _q(got[s][0]), _q(got[s][1]), _q_gust(got[s][2])
                gok.append(True)
            else:
                j = run["steps"].index(s)
                u[i], v[i], g[i] = run["u"][j], run["v"][j], run["g"][j]
                gok.append(run["gok"][j])
        new = {"id": rid, "init": init, "steps": steps, "u": u, "v": v, "g": g, "gok": gok,
               "complete": all(s in steps for s in planned) and all(gok)}
        _runs[model][rid] = new
        _series_cache.pop(model, None)
    _save(model, new)
    print(f"[wind_field] {model} {rid}: +{len(got)} steps ({sum(gok)}/{len(planned)}"
          f"{', complete' if new['complete'] else ''}) in {time.monotonic() - t0:.0f}s", flush=True)
    return True


def update(model: str, pause: float = 0.15) -> bool:
    """One pass: fetch or top up the two newest cycles that should be
    published by now (the newest is usually still publishing; the one
    before it supplies the full ten-day tail)."""
    now = datetime.now(timezone.utc)
    changed = False
    for init in _candidates(model, now)[:2]:
        run = _runs[model].get(_run_id(init))
        if run and run["complete"]:
            continue
        changed |= _fetch_run(model, init, pause)
    _prune(model)
    _status[model] = {"checked": now.isoformat(timespec="seconds"),
                      "runs": {rid: {"steps": len(r["steps"]), "complete": r["complete"]}
                               for rid, r in sorted(_runs[model].items())}}
    return changed


# ── composite series ─────────────────────────────────────────────────────────

def _encode(a: np.ndarray) -> bytes:
    """Wire format for one field: int16 x-deltas within each row (first
    column raw), the high bytes of every value then the low bytes. The
    client prefix-sums each row back; int16 wraparound cancels exactly."""
    g = a.reshape(NY, NX).astype(np.int32)
    d = g.copy()
    d[:, 1:] -= g[:, :-1]
    b = d.astype(np.int16).ravel().view(np.uint8).reshape(-1, 2)   # little-endian: lo, hi
    return b[:, 1].tobytes() + b[:, 0].tobytes()


def _local_iso(t: datetime) -> str:
    return t.astimezone(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%dT%H:%M")


def series(model: str) -> dict | None:
    """The map's time series: for every valid hour within [now−3 d, newest
    run +240 h] the newest run that has it. Past hours and hours beyond +48 h
    are kept 3-hourly (the client interpolates between steps); the first 48 h
    keep the model's own cadence. Cached until a run changes."""
    with _lock:
        cached = _series_cache.get(model)
        if cached and time.monotonic() - cached["built_mono"] < 3600:
            return cached
        runs = sorted(_runs[model].values(), key=lambda r: r["init"], reverse=True)
        if not runs:
            return None
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        t_lo, t_hi = now - timedelta(days=3), runs[0]["init"] + timedelta(hours=240)
        chosen: dict[datetime, tuple] = {}
        for run in runs:                               # newest first → first writer wins
            for i, s in enumerate(run["steps"]):
                t = run["init"] + timedelta(hours=s)
                if t < t_lo or t > t_hi or t in chosen:
                    continue
                h_ahead = (t - now).total_seconds() / 3600
                if (h_ahead < -1 or h_ahead > 48) and (t.hour % 3):
                    continue
                chosen[t] = (run, i)
        times = sorted(chosen)
        if not times:
            return None
        n = len(times)
        u = np.empty((n, NY * NX), np.int16)
        v = np.empty((n, NY * NX), np.int16)
        g = np.empty((n, NY * NX), np.int16)
        steps = []
        for k, t in enumerate(times):
            run, i = chosen[t]
            u[k] = run["u"][i].ravel()
            v[k] = run["v"][i].ravel()
            g[k] = run["g"][i].ravel()
            steps.append({"t": _local_iso(t), "ms": int(t.timestamp() * 1000), "run": run["id"],
                          "lead": int(run["steps"][i])})
        now_i = min(range(n), key=lambda k: abs(times[k] - now))
        # chunks: A = now−3 h … now+48 h, B = the tail, C = the past
        a0 = max(0, now_i - 3)
        a1 = next((k for k in range(a0, n) if (times[k] - now) > timedelta(hours=48)), n)
        bounds = [(a0, a1)]
        if a1 < n:
            bounds.append((a1, n))
        if a0 > 0:
            bounds.append((0, a0))
        chunks = []
        for i0, i1 in bounds:
            raw = b"".join(_encode(u[k]) + _encode(v[k]) for k in range(i0, i1))
            chunks.append({"i0": i0, "i1": i1, "gz": gzip.compress(raw, 6)})
        # content token: which run supplies each step — a top-up that only
        # re-sources hours the previous cycle already covered leaves n alone
        tag = hashlib.blake2b(json.dumps([[s["run"], s["lead"]] for s in steps]).encode(),
                              digest_size=4).hexdigest()
        sid = f"{model}-{runs[0]['id']}-{n}-{now:%Y%m%d%H}-{tag}"
        meta = {"model": model, "grid": GRID, "scale": SCALE, "series": sid,
                "steps": steps, "now_index": now_i,
                "chunks": [{"i0": c["i0"], "i1": c["i1"], "bytes": len(c["gz"])} for c in chunks],
                "runs": sorted({s["run"] for s in steps}),
                "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        out = {"meta": meta, "chunks": chunks, "u": u, "v": v, "g": g,
               "ms": np.asarray([s["ms"] for s in steps], np.int64), "built_mono": time.monotonic()}
        _series_cache[model] = out
        return out


def step_gz(model: str, i: int, s: dict | None = None) -> bytes | None:
    s = s or series(model)
    if not s or not (0 <= i < len(s["meta"]["steps"])):
        return None
    return gzip.compress(_encode(s["u"][i]) + _encode(s["v"][i]), 6)


def sample(model: str, points: list[tuple[float, float]], times_ms) -> dict | None:
    """The series read at `points` ((lat, lon) …) for epoch-ms `times_ms`:
    exactly what the page draws at that pixel — u and v linear between the
    two steps bracketing the time, bilinear in the grid, speed from the
    interpolated components. Returns {series, speed_ms, dir_deg, gust_ms}
    (arrays [len(times), len(points)], NaN outside the series' window or the
    grid; gust NaN where the step has none) or None without a series."""
    s = series(model)
    if not s:
        return None
    t = np.asarray(times_ms, np.int64)
    ms = s["ms"]
    nt, npt = len(t), len(points)
    out = {"series": s["meta"]["series"],
           "speed_ms": np.full((nt, npt), np.nan), "dir_deg": np.full((nt, npt), np.nan),
           "gust_ms": np.full((nt, npt), np.nan)}
    if not nt or not npt:
        return out
    inside = (t >= ms[0]) & (t <= ms[-1])
    if not inside.any():
        return out
    tt = t[inside]
    b = np.clip(np.searchsorted(ms, tt, side="right"), 1, len(ms) - 1)
    a = b - 1
    span = (ms[b] - ms[a]).astype(float)
    f = np.where(span > 0, (tt - ms[a]) / np.where(span > 0, span, 1), 0.0)   # [nt_in]
    lat = np.asarray([p[0] for p in points], float)
    lon = np.asarray([p[1] for p in points], float)
    fy, fx = (LAT_N - lat) / STEP, (lon - LON_W) / STEP
    ok = (fy >= 0) & (fy <= NY - 1) & (fx >= 0) & (fx <= NX - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, NY - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, NX - 2)
    dy, dx = fy - y0, fx - x0
    k00 = y0 * NX + x0
    corners = np.stack([k00, k00 + 1, k00 + NX, k00 + NX + 1], 1)          # [npt, 4]
    w = np.stack([(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx], 1)   # [npt, 4]

    def _bilinear(field):                                                # [nt_in, npt]
        ca = field[a][:, corners].astype(float) / SCALE                  # [nt_in, npt, 4]
        cb = field[b][:, corners].astype(float) / SCALE
        c = ca * (1 - f)[:, None, None] + cb * f[:, None, None]
        return (c * w[None]).sum(2)

    u, v = _bilinear(s["u"]), _bilinear(s["v"])
    spd = np.hypot(u, v)
    dirn = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    # gust: a step without one contributes nothing; both missing → NaN
    ga = s["g"][a][:, corners].astype(float)
    gb = s["g"][b][:, corners].astype(float)
    ga_ok, gb_ok = ga[:, :, 0] >= 0, gb[:, :, 0] >= 0                     # per step, grid-wide
    fa = np.where(gb_ok, 1 - f[:, None], 1.0) * ga_ok
    fb = np.where(ga_ok, f[:, None], 1.0) * gb_ok
    gust = ((np.maximum(ga, 0) / SCALE * w[None]).sum(2) * fa
            + (np.maximum(gb, 0) / SCALE * w[None]).sum(2) * fb)
    gust = np.where(ga_ok | gb_ok, np.maximum(gust, spd), np.nan)
    spd[:, ~ok] = np.nan; dirn[:, ~ok] = np.nan; gust[:, ~ok] = np.nan
    out["speed_ms"][inside] = spd
    out["dir_deg"][inside] = dirn
    out["gust_ms"][inside] = gust
    return out


def status() -> dict:
    return {m: {**_status.get(m, {}), "series": (series(m) or {}).get("meta", {}).get("series")}
            for m in ("GFS", "EURO")}


# ── updater thread ───────────────────────────────────────────────────────────
_UPDATE_INTERVAL = 600


def _loop():
    for m in ("GFS", "EURO"):
        _load_disk(m)
    for m in ("GFS", "EURO"):
        n = sum(len(r["steps"]) for r in _runs[m].values())
        print(f"[wind_field] {m}: {len(_runs[m])} runs / {n} steps on disk", flush=True)
    while True:
        for m in ("GFS", "EURO"):
            try:
                # data.ecmwf.int answers a burst with 429 (and the S3 mirror
                # with 503 Slow Down); three requests a step, one step a second
                update(m, pause=1.0 if m == "EURO" else 0.15)
            except Exception as e:
                print(f"[wind_field] {m} update failed: {e}", flush=True)
        time.sleep(_UPDATE_INTERVAL)


def start_updater():
    threading.Thread(target=_loop, daemon=True, name="wind-field").start()


if __name__ == "__main__":
    import sys
    for m in ("GFS", "EURO"):
        _load_disk(m)
    models = sys.argv[1:] or ["GFS", "EURO"]
    for m in models:
        t0 = time.time()
        print(m, "changed" if update(m) else "no change", f"{time.time() - t0:.0f}s")
        s = series(m)
        if s:
            md = s["meta"]
            print(f"  series {md['series']}: {len(md['steps'])} steps, "
                  f"{md['steps'][0]['t']} → {md['steps'][-1]['t']}, "
                  f"chunks {[c['bytes'] for c in md['chunks']]}")
