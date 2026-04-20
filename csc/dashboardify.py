"""CSC dashboardify — canonical transformer from raw Open-Meteo Marine API
hourly records to the exact record shape the colesurfs dashboard displays.

Why this module exists
----------------------
The user-facing dashboard never shows raw Open-Meteo `wave_height` etc.
directly. Every forecast record is first run through
`waves.py::_parse_response` + `_build_components`, which:

  * prefers the swell partitions (primary/secondary/tertiary) over the
    combined `wave_height` (which is wind-chop-contaminated),
  * filters partitions with period < 6.0 s or height <= 0,
  * falls back to combined only when every partition is filtered out
    AND the combined period is >= 6.0 s,
  * reports the highest-energy surviving partition as the primary
    (top-level wave_height_ft / wave_period_s / wave_direction_deg).

Training features that feed CSC must therefore match the dashboard's
transformed values, not the raw API output. This module is the
single source of truth for that conversion.

Implementation strategy
-----------------------
Zero logic duplication: we import `_parse_response` from `waves.py`
and adapt single-record calls by wrapping the raw dict in the
column-oriented hourly block that `_parse_response` expects. Any
change to the dashboard's transformation in `waves.py` is inherited
automatically, which is the entire point.
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root is importable so `import waves` works when this
# module is imported from scripts that don't set PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from waves import _parse_response  # noqa: E402

# Full set of hourly keys _parse_response pulls. We list them explicitly
# so a raw record missing any of them is handled cleanly (filled with
# None rather than KeyError).
_HOURLY_KEYS = (
    "wave_height", "wave_period", "wave_peak_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "secondary_swell_wave_height", "secondary_swell_wave_period", "secondary_swell_wave_direction",
    "tertiary_swell_wave_height", "tertiary_swell_wave_period", "tertiary_swell_wave_direction",
)


def _raw_to_hourly_block(raw: dict) -> dict:
    """Wrap a single raw Open-Meteo hourly record into the column-oriented
    hourly block that `_parse_response` consumes."""
    time = raw.get("time")
    block = {"time": [time]}
    for k in _HOURLY_KEYS:
        block[k] = [raw.get(k)]
    return block


def dashboardify(raw: dict) -> dict:
    """Replay waves.py::_parse_response + _build_components on a single
    hourly raw Open-Meteo record.

    Returns a dict with the exact keys `waves._parse_response` emits per
    timestep:
        time:                     ISO string from the input (passthrough)
        wave_height_ft:           primary-swell partition height (ft) or None
        wave_period_s:            primary-swell partition period (s) or None
        wave_direction_deg:       primary-swell partition direction (deg) or None
        energy:                   height_ft^2 * period_s, or None
        components:               [{height_ft, period_s, direction_deg, energy,
                                    type='swell'|'swell2'|'swell3'|'combined'}, ...]
                                  Ordered by energy (highest first), filtered to
                                  period >= 6.0 s and height > 0. Wind-wave
                                  excluded. At most 2 entries (matches
                                  _build_components' comps[:2]).
        raw_direction_deg:        best-available swell direction, even when
                                  no partition survives filtering (for the map arrow)
        combined_wave_height_m:   raw combined Hs (m), passthrough
        combined_wave_period_s:   peak period if present, else mean period (s)
        combined_wave_direction_deg: raw combined direction (deg)

    This output is bit-for-bit identical to what waves.py emits in a
    forecast record — the dashboard's source of truth for GFS/EURO cells.
    """
    block = _raw_to_hourly_block(raw)
    records = _parse_response({"hourly": block})
    return records[0]


def dashboardify_series(hourly: dict) -> list[dict]:
    """Apply `dashboardify` across every timestep in an Open-Meteo
    `data['hourly']` block (column-oriented lists). Returns a list of
    per-hour dashboard records in the same order as hourly['time'],
    matching the shape of `waves._parse_response`'s return value.
    """
    return _parse_response({"hourly": hourly})


# ─── Verification (run as __main__) ───────────────────────────────────────

def _reconstruct_raw_from_parquet_rows(rows) -> dict:
    """Given a pandas DataFrame slice with (variable, value) long rows for
    a single (buoy, model, valid_utc) triple, rebuild the raw Open-Meteo
    hourly record dict keyed by Open-Meteo variable name."""
    raw = {}
    for _, r in rows.iterrows():
        raw[r["variable"]] = float(r["value"]) if r["value"] is not None else None
    return raw


def _verify() -> int:
    """Load 5 random (buoy, hour) triples from the GFS forecast archive,
    reconstruct the raw record, run `dashboardify`, fetch the matching
    record from the live /api/forecast/GFS endpoint, and assert equality.
    Returns 0 on full success, 1 on any mismatch or fatal setup error.
    """
    import random
    import pandas as pd
    import requests

    from csc.schema import FORECASTS_DIR

    # Find shards under model=GFS across all buoys and months.
    shards = sorted((FORECASTS_DIR / "model=GFS").rglob("*.parquet"))
    if not shards:
        print("[verify] no GFS shards under .csc_data/forecasts/ — cannot verify")
        return 1

    # Read all of them — they're small. We'll sample 5 random rows.
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df = df[df["lead_days"] == 0]
    if df.empty:
        print("[verify] no lead_days=0 analysis rows in the archive")
        return 1

    # Group by (buoy_id, valid_utc) so we can recover full raw records.
    keys = df[["buoy_id", "valid_utc"]].drop_duplicates()
    if len(keys) < 5:
        print(f"[verify] only {len(keys)} unique (buoy, valid_utc) rows — "
              f"sampling everything")
        sample = keys
    else:
        sample = keys.sample(n=5, random_state=42)

    # Live endpoint: hit once per unique buoy_id.
    try:
        from config import SPOTS
    except Exception as e:
        print(f"[verify] cannot import SPOTS from config: {e}")
        return 1
    id_to_spot = {s["buoy_id"]: s["name"] for s in SPOTS}

    endpoint_cache: dict[str, dict] = {}

    def _fetch_live(model_key: str) -> dict | None:
        if model_key in endpoint_cache:
            return endpoint_cache[model_key]
        url = f"http://localhost:5151/api/forecast/{model_key}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[verify] live endpoint fetch failed: {exc}")
            endpoint_cache[model_key] = None
            return None
        endpoint_cache[model_key] = data
        return data

    live_by_model = _fetch_live("GFS") or {}
    if not live_by_model:
        print("[verify] live GFS endpoint returned nothing — is the server up?")
        return 1

    passed = 0
    failed = 0
    for _, key in sample.iterrows():
        buoy_id = key["buoy_id"]
        valid_utc = pd.Timestamp(key["valid_utc"])
        rows = df[(df["buoy_id"] == buoy_id) & (df["valid_utc"] == valid_utc)]
        raw = _reconstruct_raw_from_parquet_rows(rows)
        # _parse_response uses the 'time' key as a passthrough string.
        # The live endpoint's time is the Open-Meteo "timezone=auto-or-UTC"
        # ISO string without offset suffix, matching what we pass in.
        raw["time"] = valid_utc.strftime("%Y-%m-%dT%H:%M")

        ours = dashboardify(raw)

        spot_name = id_to_spot.get(buoy_id)
        if not spot_name:
            print(f"[verify] buoy_id {buoy_id} not in SPOTS — skipping")
            continue
        live_records = live_by_model.get(spot_name) or []
        # Find by time prefix (Open-Meteo sometimes emits :00, sometimes not)
        target_prefix = valid_utc.strftime("%Y-%m-%dT%H:")
        live_rec = next(
            (r for r in live_records if r.get("time", "").startswith(target_prefix)),
            None,
        )
        if live_rec is None:
            print(f"[verify] {buoy_id}/{spot_name} @ {target_prefix} "
                  f"not present in live feed (forecast horizon?) — skipping")
            continue

        # Field-by-field compare. We ignore `time` string formatting and
        # `combined_wave_*` passthroughs (which may differ due to server-side
        # caching windows vs. the archive). The load-bearing identity is
        # the primary + components.
        fields = ("wave_height_ft", "wave_period_s", "wave_direction_deg",
                  "energy", "components", "raw_direction_deg")
        mismatches = []
        for f in fields:
            if ours.get(f) != live_rec.get(f):
                mismatches.append((f, ours.get(f), live_rec.get(f)))
        if mismatches:
            failed += 1
            print(f"[verify] MISMATCH {buoy_id}/{spot_name} @ {target_prefix}")
            for f, a, b in mismatches:
                print(f"    {f}: ours={a!r}  live={b!r}")
        else:
            passed += 1
            print(f"[verify] OK      {buoy_id}/{spot_name} @ {target_prefix}")

    print(f"[verify] {passed} passed, {failed} failed, "
          f"{len(sample) - passed - failed} skipped")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_verify())
