"""CSC — CMEMS partitioned WAM backfill.

Pull Copernicus Marine Service (`marine.copernicus.eu`) ECMWF-WAM output
WITH swell partitions for each CSC buoy point, for a requested year range,
and emit per-(buoy, year) Parquet shards under

    .csc_data/euro_partitions/buoy=<id>/year=<y>/cmems.parquet

Why this module exists
----------------------
Open-Meteo's `ecmwf_wam025` endpoint exposes only four *combined* variables
(wave_height, wave_period, wave_peak_period, wave_direction) — all
bulk-sea-state including wind chop. CMEMS, which ingests the same ECMWF
WAM output, publishes the partitioned fields we actually need:

  VHM0        total sig wave height (matches Open-Meteo wave_height)
  VHM0_SW1    primary swell Hs                  ← critical for CSC EURO features
  VHM0_SW2    secondary swell Hs
  VHM0_WW    wind-sea Hs (diagnostic only; not written)
  VTM01_SW1   primary swell mean period
  VTM01_SW2   secondary swell mean period
  VMDR_SW1    primary swell mean-from direction
  VMDR_SW2    secondary swell mean-from direction
  VTPK        spectral peak period of total sea state
  VMDR        mean-from direction of total sea state

Product selection
-----------------
We hit two CMEMS products, chosen for multi-year coverage + partition
fields + free access + point-extract-friendly:

  1. `cmems_mod_glo_wav_my_0.2deg_PT3H-i`           (reanalysis, 1993-now-ish, 3-hourly, 0.2°)
  2. `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i`       (operational analysis+forecast, 0.083°, 3-hourly)

The cutover from MY → ANFC happens at the point where MY stops being
produced (~6–12 months behind real time). The backfill prefers MY where
available and falls back to ANFC for recent dates.

Auth
----
The `copernicusmarine` Python package looks for credentials in, in order:
  1. env vars COPERNICUS_MARINE_SERVICE_USERNAME / _PASSWORD
  2. ~/.copernicusmarine/credentials.txt (written by `copernicusmarine login`)

Sign up (free) at https://marine.copernicus.eu/ (one-time, ~2 minutes)
then run `copernicusmarine login` once to persist credentials.

Output schema
-------------
Columns written (same names the Open-Meteo loader uses, so downstream
feature code can treat CMEMS and OM as drop-in replacements):

  buoy_id                          str      e.g. "44097"
  valid_utc                        str (ISO 8601, UTC)
  swell_wave_height                float    meters  (VHM0_SW1)
  swell_wave_period                float    seconds (VTM01_SW1)
  swell_wave_direction             float    deg FROM (VMDR_SW1)
  secondary_swell_wave_height      float    meters  (VHM0_SW2)
  secondary_swell_wave_period      float    seconds (VTM01_SW2)
  secondary_swell_wave_direction   float    deg FROM (VMDR_SW2)
  wave_height                      float    meters  (VHM0)
  wave_period                      float    seconds (VTPK)
  wave_direction                   float    deg FROM (VMDR)
  source                           str      "cmems_wam_my" | "cmems_wam_anfc"
  product                          str      CMEMS dataset id
  ingest_utc                       str (ISO 8601)

Run
---
  python -m csc.cmems_backfill                      # all 8 buoys, 2017..now
  python -m csc.cmems_backfill --buoy 44097         # one buoy
  python -m csc.cmems_backfill --years 2023 2024    # override window
  python -m csc.cmems_backfill --buoy 44097 --years 2024 --force    # overwrite

Resumability
------------
Per-(buoy, year) shards that already exist are skipped. Pass --force to
re-download. The MY / ANFC split is handled per-year: if a year sits on
the boundary and both products return data, rows are concatenated and
de-duped on valid_utc (MY wins over ANFC where they overlap).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from csc.schema import BUOYS, CSC_DATA_DIR, ensure_dirs

# ─── Config ───────────────────────────────────────────────────────────────

EURO_PART_DIR = CSC_DATA_DIR / "euro_partitions"

# CMEMS product ids. See https://data.marine.copernicus.eu/products for
# the public catalog. Both products expose the same VHM0_SW1/SW2/WW
# partition variables.
CMEMS_PRODUCT_MY = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
CMEMS_PRODUCT_ANFC = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"

# Variables to request from CMEMS. Short-circuits the full 1.5 TB global
# grid — copernicusmarine subsets server-side.
CMEMS_VARS = [
    "VHM0",       # total Hs
    "VTPK",       # total Tp
    "VMDR",       # total mean dir
    "VHM0_SW1",   # primary swell Hs
    "VTM01_SW1",  # primary swell mean period
    "VMDR_SW1",   # primary swell mean dir
    "VHM0_SW2",   # secondary swell Hs
    "VTM01_SW2",  # secondary swell mean period
    "VMDR_SW2",   # secondary swell mean dir
]

# Column remap from CMEMS → CSC canonical names.
CMEMS_TO_CANONICAL = {
    "VHM0":      "wave_height",
    "VTPK":      "wave_period",
    "VMDR":      "wave_direction",
    "VHM0_SW1":  "swell_wave_height",
    "VTM01_SW1": "swell_wave_period",
    "VMDR_SW1":  "swell_wave_direction",
    "VHM0_SW2":  "secondary_swell_wave_height",
    "VTM01_SW2": "secondary_swell_wave_period",
    "VMDR_SW2":  "secondary_swell_wave_direction",
}

# CSC buoys sit at these points. copernicusmarine does a nearest-grid-cell
# lookup server-side when we pass a single-point bbox.
def _buoy_points() -> dict[str, tuple[float, float]]:
    return {b[0]: (float(b[2]), float(b[3])) for b in BUOYS}


# ─── Shard layout ─────────────────────────────────────────────────────────

def _shard_path(buoy_id: str, year: int) -> Path:
    return (EURO_PART_DIR / f"buoy={buoy_id}" / f"year={year}"
            / "cmems.parquet")


def _shard_exists(buoy_id: str, year: int) -> bool:
    p = _shard_path(buoy_id, year)
    return p.exists() and p.stat().st_size > 0


# ─── CMEMS fetch ──────────────────────────────────────────────────────────

def _import_copernicusmarine():
    """Lazy-import the copernicusmarine package, producing a clear error
    message if it's not installed."""
    try:
        import copernicusmarine  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "copernicusmarine not installed. Run:\n"
            "  pip install copernicusmarine\n"
            "inside the colesurfs conda env, then `copernicusmarine login`."
        ) from e
    return copernicusmarine


def _check_credentials() -> tuple[bool, str]:
    """Return (ok, message). CMEMS needs either env vars or a persisted
    credentials file."""
    if (os.environ.get("COPERNICUS_MARINE_SERVICE_USERNAME")
            and os.environ.get("COPERNICUS_MARINE_SERVICE_PASSWORD")):
        return True, "env-var credentials present"
    cred_file = Path.home() / ".copernicusmarine" / "credentials.txt"
    if cred_file.exists():
        return True, f"credentials file found at {cred_file}"
    # copernicusmarine 1.x writes under a slightly different name; check it too.
    alt = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
    if alt.exists():
        return True, f"credentials file found at {alt}"
    return False, (
        "No CMEMS credentials found. Sign up (free) at "
        "https://marine.copernicus.eu/ then run:\n"
        "  copernicusmarine login\n"
        "to persist credentials, OR export env vars "
        "COPERNICUS_MARINE_SERVICE_USERNAME / _PASSWORD."
    )


def _subset_one(
    product: str,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    variables: list[str],
) -> "pd.DataFrame":
    """Ask copernicusmarine for the 3-hourly partition fields at a single
    grid point between `start` and `end` inclusive. Returns a DataFrame
    indexed by valid_utc with one column per CMEMS variable (raw names,
    not yet canonical).

    Strategy
    --------
    `copernicusmarine.subset` returns an in-memory xarray.Dataset when
    given `return_as_xarray=True` (v2 API) / when the caller asks
    `.open_dataset(...)` in v1. Around a single grid cell this is tiny
    (a few hundred KB / year), well within RAM. We convert to a pandas
    frame with one row per timestep.
    """
    cm = _import_copernicusmarine()
    # A 0.2-degree bbox around the buoy point comfortably picks up one
    # grid cell on both MY (0.2°) and ANFC (0.083°) products. CMEMS
    # server-side selects the nearest cell; we then squeeze.
    pad = 0.12
    kwargs = dict(
        dataset_id=product,
        variables=variables,
        minimum_longitude=lon - pad,
        maximum_longitude=lon + pad,
        minimum_latitude=lat - pad,
        maximum_latitude=lat + pad,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    # copernicusmarine >=2.0 exposes `open_dataset` returning xarray.
    # Older releases (1.x) used `subset(..., return_as_xarray=True)`.
    # Try the modern API first, fall back to the 1.x API.
    ds = None
    open_ds = getattr(cm, "open_dataset", None)
    if callable(open_ds):
        ds = open_ds(**kwargs)
    else:
        subset_fn = getattr(cm, "subset", None)
        if subset_fn is None:
            raise RuntimeError(
                "copernicusmarine package exposes neither open_dataset "
                "nor subset — is the installed version supported?"
            )
        try:
            ds = subset_fn(**kwargs, return_as_xarray=True)  # type: ignore[arg-type]
        except TypeError:
            # Fall back to CLI-style: force a download then open the nc.
            ds = subset_fn(**kwargs)
            # The subset() legacy path returns a Path/str to the netcdf output.
            if isinstance(ds, (str, Path)):
                import xarray as xr  # type: ignore
                ds = xr.open_dataset(str(ds))

    if ds is None:
        raise RuntimeError(f"CMEMS subset returned no dataset for {product}")

    # Squeeze lat/lon (single cell) → 1D series per variable, indexed by time.
    try:
        # Prefer .sel(method='nearest') to pin to the requested point.
        ds_pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
    except Exception:
        # Fallback: if the axis names differ, just squeeze.
        ds_pt = ds.squeeze()

    cols: dict[str, Any] = {}
    for v in variables:
        if v not in ds_pt.variables and v not in getattr(ds_pt, "data_vars", {}):
            continue
        arr = ds_pt[v].values.reshape(-1)
        cols[v] = arr
    if not cols:
        return pd.DataFrame()

    # Time axis (usually named 'time'). Convert to ISO UTC strings.
    time_name = None
    for cand in ("time", "valid_time", "valid_utc"):
        if cand in ds_pt.coords:
            time_name = cand
            break
    if time_name is None:
        raise RuntimeError(
            f"CMEMS subset for {product} has no recognizable time coord; "
            f"coords were {list(ds_pt.coords)}"
        )
    t = pd.to_datetime(ds_pt[time_name].values, utc=True)

    df = pd.DataFrame(cols)
    df.insert(0, "valid_utc", t.strftime("%Y-%m-%dT%H:%M:%S%z"))
    # Drop rows where ALL partition vars are NaN (outside product window).
    core_cols = [c for c in df.columns if c != "valid_utc"]
    df = df.dropna(axis=0, how="all", subset=core_cols)
    return df


def _canonicalize(df: pd.DataFrame, buoy_id: str, source: str,
                  product: str) -> pd.DataFrame:
    """Remap CMEMS variable names to CSC canonical column names, attach
    identity columns, and enforce column order."""
    if df.empty:
        return df
    out = df.rename(columns=CMEMS_TO_CANONICAL).copy()
    out.insert(0, "buoy_id", buoy_id)
    out["source"] = source
    out["product"] = product
    out["ingest_utc"] = datetime.now(timezone.utc).isoformat()
    cols = [
        "buoy_id", "valid_utc",
        "swell_wave_height", "swell_wave_period", "swell_wave_direction",
        "secondary_swell_wave_height", "secondary_swell_wave_period",
        "secondary_swell_wave_direction",
        "wave_height", "wave_period", "wave_direction",
        "source", "product", "ingest_utc",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


def _fetch_buoy_year(
    buoy_id: str,
    lat: float,
    lon: float,
    year: int,
) -> pd.DataFrame:
    """Pull one (buoy, year) slice from CMEMS. Tries MY reanalysis first
    (better quality, longer record), then ANFC for the most recent window
    where MY hasn't caught up. Returns a canonicalized frame, possibly
    empty."""
    start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    # End-exclusive; CMEMS is inclusive so subtract 3 hours (the 3-hourly
    # grid's native step) to avoid double-writing a year boundary.
    end = datetime(year, 12, 31, 21, 0, 0, tzinfo=timezone.utc)

    frames: list[pd.DataFrame] = []

    # Try MY reanalysis first.
    try:
        print(f"    → {CMEMS_PRODUCT_MY} @ ({lat:.3f}, {lon:.3f}) {year}")
        my_df = _subset_one(CMEMS_PRODUCT_MY, lat, lon, start, end, CMEMS_VARS)
        if not my_df.empty:
            frames.append(_canonicalize(my_df, buoy_id, "cmems_wam_my",
                                        CMEMS_PRODUCT_MY))
            print(f"      MY: {len(my_df)} rows")
    except Exception as e:
        # MY is expected to 404 for the current year; don't noisily warn then.
        if year >= datetime.now(timezone.utc).year - 1:
            print(f"      MY: unavailable ({type(e).__name__}), falling back to ANFC")
        else:
            print(f"      MY failed for {year}: {e}", file=sys.stderr)

    # Fill gaps with ANFC. For years where MY returned a full record we
    # still try ANFC — but only if MY came back empty or very thin. This
    # keeps us from double-charging the CMEMS portal for a full year of
    # identical hours.
    my_rows = sum(len(f) for f in frames)
    expected_rows = 366 * 24 // 3  # 3-hourly, generous upper bound
    if my_rows < expected_rows * 0.5:
        try:
            print(f"    → {CMEMS_PRODUCT_ANFC} @ ({lat:.3f}, {lon:.3f}) {year}")
            anfc_df = _subset_one(CMEMS_PRODUCT_ANFC, lat, lon, start, end,
                                  CMEMS_VARS)
            if not anfc_df.empty:
                frames.append(_canonicalize(anfc_df, buoy_id, "cmems_wam_anfc",
                                            CMEMS_PRODUCT_ANFC))
                print(f"      ANFC: {len(anfc_df)} rows")
        except Exception as e:
            print(f"      ANFC failed for {year}: {e}", file=sys.stderr)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # De-dupe on valid_utc, preferring MY (first in list) over ANFC.
    combined = combined.drop_duplicates(subset=["valid_utc"], keep="first")
    combined = combined.sort_values("valid_utc").reset_index(drop=True)
    return combined


def _write_shard(buoy_id: str, year: int, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    path = _shard_path(buoy_id, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    return len(df)


# ─── Orchestration ────────────────────────────────────────────────────────

def _default_years() -> list[int]:
    today = datetime.now(timezone.utc).date()
    return list(range(2017, today.year + 1))


def run_backfill(
    buoy_ids: list[str] | None = None,
    years: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Pull CMEMS partitioned WAM for each (buoy, year) and shard to disk.

    Returns a report dict with per-buoy, per-year row counts.
    """
    ensure_dirs()
    EURO_PART_DIR.mkdir(parents=True, exist_ok=True)

    ok, msg = _check_credentials()
    print(f"CMEMS credentials: {msg}")
    if not ok:
        print("\n=== CMEMS BACKFILL ABORTED — CREDENTIALS REQUIRED ===")
        print("To get access (free, takes ~2 minutes):")
        print("  1. Sign up at https://marine.copernicus.eu/ "
              "(click 'Register' top right)")
        print("  2. Verify your email, then come back to this shell")
        print("  3. Run:  copernicusmarine login")
        print("     and enter the username + password you just registered")
        print("Then re-run this module.\n")
        return {
            "aborted": True,
            "reason": "no_credentials",
            "signup_url": "https://marine.copernicus.eu/",
        }

    # Fail early if the package itself is missing, before we start looping.
    _import_copernicusmarine()

    points = _buoy_points()
    target_ids = buoy_ids or [b[0] for b in BUOYS]
    target_years = years or _default_years()

    coverage: dict[str, dict[int, int]] = {bid: {} for bid in target_ids}
    warnings: list[str] = []

    for bid in target_ids:
        if bid not in points:
            warnings.append(f"{bid}: not in BUOYS table, skipping")
            continue
        lat, lon = points[bid]
        meta = next((b for b in BUOYS if b[0] == bid), None)
        label = meta[1] if meta else "?"
        print(f"\n── {bid}  {label}  ({lat:.3f}, {lon:.3f}) ──", flush=True)

        for y in target_years:
            if not force and _shard_exists(bid, y):
                # Inspect existing row count so coverage reporting is accurate.
                try:
                    df_existing = pd.read_parquet(_shard_path(bid, y))
                    coverage[bid][y] = len(df_existing)
                    print(f"  {bid} {y}: {len(df_existing):>6} rows (shard exists, skipping)")
                except Exception:
                    coverage[bid][y] = 0
                    print(f"  {bid} {y}: shard exists but unreadable, skipping")
                continue
            print(f"  {bid} {y}: fetching…")
            try:
                df = _fetch_buoy_year(bid, lat, lon, y)
            except Exception:
                traceback.print_exc()
                df = pd.DataFrame()
            n = _write_shard(bid, y, df)
            coverage[bid][y] = n
            print(f"  {bid} {y}: {n:>6} rows written")

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
    print("\n─────────── CMEMS partitioned-WAM backfill coverage ───────────")
    header = f"{'buoy':<6}  " + "  ".join(f"{y:>6}" for y in years) + "    total"
    print(header)
    print("-" * len(header))
    for bid, by_year in coverage.items():
        total = sum(by_year.values())
        cells = "  ".join(f"{by_year.get(y, 0):>6}" for y in years)
        print(f"{bid:<6}  {cells}    {total:>6}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.cmems_backfill")
    ap.add_argument("--buoy", default=None,
                    help="Limit to one buoy id (default: all 8 CSC buoys).")
    ap.add_argument("--years", nargs="*", type=int, default=None,
                    help="Year list (default: 2017..current).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing shards (default: skip).")
    args = ap.parse_args()
    buoy_ids = [args.buoy] if args.buoy else None
    report = run_backfill(buoy_ids=buoy_ids, years=args.years, force=args.force)
    return 1 if report.get("aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
