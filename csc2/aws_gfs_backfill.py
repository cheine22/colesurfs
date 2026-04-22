"""CSC2 — NOAA GFS-Wave forecast backfill via AWS Open Data.

Sources lead-time-resolved GFS-Wave forecasts from the public S3 bucket
    s3://noaa-gfs-bdp-pds/gfs.{YYYYMMDD}/{HH}/wave/gridded/
        gfswave.t{HH}z.global.0p25.f{NNN}.grib2
using byte-range requests driven by the GRIB2 `.idx` sidecar files. Only the
swell-partition messages we actually need are pulled — SHTS (Hs), MPTS (Tp),
SWDIR — at each of the three partition levels (level=1,2,3 = primary,
secondary, tertiary). Everything else stays in S3.

For each cycle × buoy, extracted rows go through the same processing as the
live logger: waves.py's `_parse_response`-equivalent logic (5.0 s period
filter, energy-sort top-2, no combined fallback) via `records_to_rows`
from csc2.logger. Output shards are byte-compatible with live-logger
output at `.csc2_data/forecasts/model=GFS/buoy=.../cycle=.parquet`.

Usage:
    python -m csc2.aws_gfs_backfill                      # defaults (see ap)
    python -m csc2.aws_gfs_backfill --start 2025-04-28   # custom start date
    python -m csc2.aws_gfs_backfill --cycles 00,12       # which UTC cycles
    python -m csc2.aws_gfs_backfill --lead-step 3        # hour spacing
    python -m csc2.aws_gfs_backfill --buoy 44065 --end 2025-05-01  # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone as dtz
from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csc2.schema import BUOYS, FORECASTS_DIR, LOGS_DIR, ensure_dirs  # noqa: E402
from csc2.logger import shard_path, records_to_rows, write_rows  # noqa: E402
from config import m_to_ft  # noqa: E402


S3_BUCKET = "noaa-gfs-bdp-pds"
GRID = "global.0p25"

# NOAA `.idx` sidecar short names differ from the eccodes GRIB2 shortNames.
# Observed canonical pattern for gfswave.t{HH}z.global.0p25.f{NNN}.grib2.idx:
#   SWELL:1 in sequence  →  eccodes shortName 'shts' at level=1 (swell Hs)
#   SWPER:1 in sequence  →  eccodes shortName 'mpts' at level=1 (swell Tp)
#   SWDIR:1 in sequence  →  eccodes shortName 'swdir' at level=1 (swell Dir)
# We pull all three variables × levels 1/2/3 per lead file.
IDX_SHORT_WANTED = {"SWELL", "SWPER", "SWDIR"}
IDX_LEVEL_WANTED = {"1", "2", "3"}

# eccodes → canonical key used downstream (after GRIB parse).
ECCODES_SHORT_TO_ROLE = {"shts": "HT", "mpts": "PR", "swdir": "DR"}


def _s3() -> "boto3.client":
    return boto3.client("s3",
                        config=Config(signature_version=UNSIGNED,
                                      region_name="us-east-1",
                                      retries={"max_attempts": 3, "mode": "standard"}))


def _grib_key(cycle_dt: datetime, lead_h: int) -> str:
    """S3 key for gfswave.tHHz.global.0p25.fNNN.grib2."""
    ymd = cycle_dt.strftime("%Y%m%d")
    hh = cycle_dt.strftime("%H")
    nnn = f"{lead_h:03d}"
    return f"gfs.{ymd}/{hh}/wave/{('gridded' if True else '')}/gfswave.t{hh}z.{GRID}.f{nnn}.grib2"


def _fetch_idx(s3, key: str) -> list[tuple[int, int, str, str]]:
    """Parse the `.idx` sidecar. Returns list of (byte_start, byte_end, short, level)
    for messages whose short is in IDX_SHORT_WANTED and whose level starts with
    '1 in', '2 in', or '3 in' (partition 1/2/3). Ordered by byte offset."""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key + ".idx")
    except Exception:
        return []
    body = obj["Body"].read().decode("ascii", errors="ignore")
    rows: list[tuple[int, str, str]] = []
    for ln in body.splitlines():
        parts = ln.split(":")
        if len(parts) < 6:
            continue
        try:
            byte_start = int(parts[1])
        except ValueError:
            continue
        rows.append((byte_start, parts[3], parts[4]))
    rows.sort(key=lambda x: x[0])
    out: list[tuple[int, int, str, str]] = []
    for i, (start, short, level_tok) in enumerate(rows):
        end = rows[i + 1][0] - 1 if i + 1 < len(rows) else None
        if short not in IDX_SHORT_WANTED:
            continue
        # "1 in sequence (insta)" → "1"
        level_n = level_tok.split(" ", 1)[0]
        if level_n not in IDX_LEVEL_WANTED:
            continue
        out.append((start, end, short, level_n))
    return out


def _download_partial(s3, key: str, ranges: list[tuple[int, int, str, str]]) -> Path | None:
    """Fetch just the byte ranges containing partition messages. Writes a
    .grib2 fragment to a temp file; returns its path (caller must delete)."""
    if not ranges:
        return None
    tmp = NamedTemporaryFile(suffix=".grib2", delete=False)
    tmp_path = Path(tmp.name)
    try:
        for start, end, _short, _level in ranges:
            # Range header; if end is None, open-ended to EOF.
            range_header = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key, Range=range_header)
            tmp.write(obj["Body"].read())
        tmp.close()
        return tmp_path
    except Exception:
        tmp.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _extract_points(grib_path: Path,
                     buoys: list[tuple[str, float, float]]) -> dict[str, dict[tuple[str, str], float | None]]:
    """Open the partial GRIB and extract the 9 partition values at each buoy.
    Returns {buoy_id: {("HT"|"PR"|"DR", "1"|"2"|"3"): value}}."""
    import eccodes as ec
    out: dict[str, dict[tuple[str, str], float | None]] = {b[0]: {} for b in buoys}
    with grib_path.open("rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                short = ec.codes_get(h, "shortName")
                try:
                    level = str(int(ec.codes_get(h, "level")))
                except Exception:
                    level = "0"
                role = ECCODES_SHORT_TO_ROLE.get(short.lower())
                if role is None or level not in IDX_LEVEL_WANTED:
                    continue
                for buoy_id, lat, lon in buoys:
                    lon360 = lon % 360
                    try:
                        nearest = ec.codes_grib_find_nearest(h, lat, lon360, is_lsm=0, npoints=1)
                        val = float(nearest[0]["value"]) if nearest else None
                        if val is not None and (val == 9999.0 or val != val):
                            val = None
                    except Exception:
                        val = None
                    out[buoy_id][(role, level)] = val
            finally:
                ec.codes_release(h)
    return out


def _raw_row(utc: datetime, extracted: dict[tuple[str, str], float | None]) -> dict:
    """Flatten a per-buoy extracted dict into the row shape that
    waves._build_components consumes."""
    def g(role, lvl):
        return extracted.get((role, str(lvl)))
    return {
        "time": utc.strftime("%Y-%m-%dT%H:%M"),
        "sw1_h_m": g("HT", 1), "sw1_p_s": g("PR", 1), "sw1_d": g("DR", 1),
        "sw2_h_m": g("HT", 2), "sw2_p_s": g("PR", 2), "sw2_d": g("DR", 2),
        "sw3_h_m": g("HT", 3), "sw3_p_s": g("PR", 3), "sw3_d": g("DR", 3),
    }


def _raw_rows_to_records(rows: list[dict]) -> list[dict]:
    """Apply the same post-processing as waves._build_components so the
    stored partition pair is identical to the live-logger output."""
    from waves import _build_components, _safe  # noqa: E402

    out = []
    for r in rows:
        comps = _build_components(
            r.get("sw1_h_m"), r.get("sw1_p_s"), r.get("sw1_d"),
            r.get("sw2_h_m"), r.get("sw2_p_s"), r.get("sw2_d"),
            r.get("sw3_h_m"), r.get("sw3_p_s"), r.get("sw3_d"),
            None, None, None,
        )
        primary = comps[0] if comps else None
        out.append({
            "time":                        r["time"],
            "wave_height_ft":              primary["height_ft"] if primary else None,
            "wave_period_s":               primary["period_s"]  if primary else None,
            "wave_direction_deg":          primary["direction_deg"] if primary else None,
            "energy":                      primary["energy"] if primary else None,
            "components":                  comps,
            "raw_direction_deg":           primary["direction_deg"] if primary else None,
            "combined_wave_height_m":      None,
            "combined_wave_period_s":      None,
            "combined_wave_direction_deg": None,
        })
    return out


def _lead_steps(lead_max: int, lead_step: int) -> list[int]:
    """Lead hours we'll pull per cycle. Capped at lead_max; stepped every
    `lead_step` hours. Also always includes the pre-240 h quirks of GFS-Wave
    (hourly 0-120, 3-hourly 120-384)."""
    if lead_step < 1:
        lead_step = 3
    return list(range(0, lead_max + 1, lead_step))


def _cycles_between(start: datetime, end: datetime, hours: list[int]) -> list[datetime]:
    out: list[datetime] = []
    d = datetime(start.year, start.month, start.day, tzinfo=dtz.utc)
    while d <= end:
        for h in hours:
            c = d.replace(hour=h)
            if start <= c <= end:
                out.append(c)
        d += timedelta(days=1)
    return out


def _process_cycle(cycle_dt: datetime, *, buoy_ids: list[str],
                    buoys: list[tuple[str, float, float]],
                    lead_steps: list[int], force: bool) -> dict:
    """Download & extract a single cycle's leads; write per-buoy parquet shards.
    Returns rows-written per buoy."""
    s3 = _s3()
    cycle_id = cycle_dt.strftime("%Y%m%dT%HZ")
    ingest_utc = datetime.now(dtz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Skip whole cycle if every buoy's shard already exists.
    if not force and all(shard_path(bid, "GFS", cycle_id).exists() for bid in buoy_ids):
        return {"cycle": cycle_id, "skipped_whole": True, "rows_by_buoy": {}}

    by_buoy_rows: dict[str, list[dict]] = {bid: [] for bid in buoy_ids}
    missing_leads = 0

    for lh in lead_steps:
        key = _grib_key(cycle_dt, lh)
        ranges = _fetch_idx(s3, key)
        if not ranges:
            missing_leads += 1
            continue
        try:
            frag = _download_partial(s3, key, ranges)
        except Exception:
            missing_leads += 1
            continue
        if frag is None:
            missing_leads += 1
            continue
        try:
            extracted = _extract_points(frag, buoys)
        except Exception:
            extracted = {b[0]: {} for b in buoys}
        finally:
            try:
                frag.unlink()
            except Exception:
                pass

        valid_utc = cycle_dt + timedelta(hours=lh)
        for buoy_id, ext in extracted.items():
            row = _raw_row(valid_utc, ext)
            by_buoy_rows[buoy_id].append(row)

    # Convert raw rows → dashboard-format records → parquet-schema rows
    summary = {"cycle": cycle_id, "missing_leads": missing_leads, "rows_by_buoy": {}}
    for buoy_id, raw_rows in by_buoy_rows.items():
        raw_rows.sort(key=lambda r: r["time"])
        if not raw_rows:
            summary["rows_by_buoy"][buoy_id] = 0
            continue
        records = _raw_rows_to_records(raw_rows)
        rows = records_to_rows(records, buoy_id=buoy_id, model="GFS",
                                 cycle_utc=cycle_id, ingest_utc=ingest_utc)
        n = write_rows(buoy_id, "GFS", cycle_id, rows)
        summary["rows_by_buoy"][buoy_id] = n
    return summary


def run_backfill(*, start: datetime, end: datetime, cycle_hours: list[int],
                  lead_max: int, lead_step: int,
                  buoy_ids: list[str] | None = None,
                  parallel: int = 1, force: bool = False) -> list[dict]:
    ensure_dirs()
    targets = buoy_ids or [b[0] for b in BUOYS]
    buoys = [(bid, lat, lon) for bid, _, lat, lon, _ in BUOYS if bid in targets]
    cycles = _cycles_between(start, end, cycle_hours)
    lead_steps_list = _lead_steps(lead_max, lead_step)
    print(f"[aws_gfs_backfill] {len(cycles)} cycles × {len(lead_steps_list)} leads × "
          f"{len(buoys)} buoys — start {start} → end {end}")
    t0 = time.monotonic()
    summaries: list[dict] = []

    def _do(c):
        return _process_cycle(c, buoy_ids=targets, buoys=buoys,
                               lead_steps=lead_steps_list, force=force)

    if parallel <= 1:
        for i, c in enumerate(cycles, 1):
            try:
                s = _do(c)
            except Exception:
                traceback.print_exc()
                continue
            summaries.append(s)
            if i % 4 == 0 or i == len(cycles):
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(cycles) - i) / rate if rate > 0 else 0
                total_rows = sum(sum(s.get("rows_by_buoy", {}).values())
                                 for s in summaries)
                print(f"  {i}/{len(cycles)} cycles, {total_rows} rows, "
                      f"{rate*60:.1f} cyc/min, eta {eta/60:.1f}m", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_do, c): c for c in cycles}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    summaries.append(fut.result())
                except Exception:
                    traceback.print_exc()
                if i % 4 == 0 or i == len(cycles):
                    elapsed = time.monotonic() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(cycles) - i) / rate if rate > 0 else 0
                    total_rows = sum(sum(s.get("rows_by_buoy", {}).values())
                                     for s in summaries)
                    print(f"  {i}/{len(cycles)} cycles, {total_rows} rows, "
                          f"{rate*60:.1f} cyc/min, eta {eta/60:.1f}m", flush=True)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "aws_gfs_backfill.log").open("a") as f:
        total_rows = sum(sum(s.get("rows_by_buoy", {}).values())
                         for s in summaries)
        f.write(f"[{datetime.now(dtz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
                f"{len(summaries)} cycles, {total_rows} rows, "
                f"elapsed {time.monotonic()-t0:.0f}s\n")
    return summaries


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dtz.utc)


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc2.aws_gfs_backfill")
    ap.add_argument("--start", default="2025-04-28",
                    help="Earliest cycle date (UTC). Default matches earliest CMEMS GEE cycle.")
    ap.add_argument("--end", default=None,
                    help="Latest cycle date (UTC). Default = today.")
    ap.add_argument("--cycles", default="00,12",
                    help="UTC cycle hours per day, comma separated. Default '00,12'.")
    ap.add_argument("--lead-step", type=int, default=3,
                    help="Hour spacing of lead-time samples (default 3 — matches CMEMS ANFC).")
    ap.add_argument("--lead-max", type=int, default=240,
                    help="Max lead hour (default 240 = 10 days).")
    ap.add_argument("--buoy", default=None, help="Limit to one buoy id.")
    ap.add_argument("--parallel", type=int, default=2,
                    help="Concurrent cycle downloads (default 2). Too high → S3 throttles.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing per-cycle shards.")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else datetime.now(dtz.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    cycle_hours = [int(x) for x in args.cycles.split(",")]

    ids = [args.buoy] if args.buoy else None
    summaries = run_backfill(start=start, end=end, cycle_hours=cycle_hours,
                              lead_max=args.lead_max, lead_step=args.lead_step,
                              buoy_ids=ids, parallel=args.parallel, force=args.force)
    total_rows = sum(sum(s.get("rows_by_buoy", {}).values()) for s in summaries)
    print(f"\n[aws_gfs_backfill] done: {len(summaries)} cycles, {total_rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
