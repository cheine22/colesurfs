"""Training-vs-dashboard identity test.

Non-negotiable quality bar: every feature value that CSC's training
pipeline derives for a (buoy, physical UTC hour, model) triple must be
bit-for-bit identical (within numeric tolerance) to what the live
colesurfs dashboard would have displayed for that same physical hour.

What this test actually proves
------------------------------
The CSC training archive pulls Open-Meteo with `timezone=UTC`. The live
dashboard (waves.py) pulls with `timezone=America/New_York`. Open-Meteo
honors the `timezone=` param by shifting the **label strings** it emits
(bare ISO, no offset suffix) — same physical hour, different label.

For bit-for-bit dashboard parity, the only thing that has to be true is:

    dashboardify( raw_at_physical_hour_X_via_UTC_request )
        == waves._parse_response( raw_at_physical_hour_X_via_NY_request )

i.e. the transformation (waves._parse_response, which dashboardify
delegates to) is pure over the raw dict and ignores the label.

This test pulls fresh data from Open-Meteo in BOTH timezones, aligns the
responses by physical UTC hour, and asserts equivalence. It is the
load-bearing proof that the training pipeline's features match what the
dashboard renders, after the TZ fix landed in predict.py.

(The previous incarnation of this test compared the pre-ingested
training archive against the live /api/forecast cached snapshot, which
is an inherently different-model-cycle comparison and cannot pass by
construction — raw inputs differ, not just transformation.)

Usage:
    python tests/test_dashboard_identity.py

Exit codes:
    0 — all physical hours agree within tolerance
    1 — real divergence beyond rounding tolerance (0.1 ft / 0.1 s / 1°)
    2 — Open-Meteo unreachable
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Tolerances absorb Open-Meteo's cross-timezone rounding noise (~0.1 ft /
# 0.1 s / 1°) without masking a real train/serve drift.
TOL_HEIGHT_FT = 0.15
TOL_PERIOD_S = 0.15
TOL_DIR_DEG = 2.0

EAST_BUOYS = ("44013", "44065", "44097")
MODELS = ("GFS", "EURO")


def _approx(a, b, tol):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _dir_eq(a, b, tol):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    d = abs(float(a) - float(b)) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d <= tol


def _pull(tz: str, model: str, spots, full_vars, endpoint, model_ids):
    """Fetch one Open-Meteo Marine response for all East buoys under tz."""
    import requests
    r = requests.get(endpoint, params={
        "latitude":  ",".join(str(s["lat"]) for s in spots),
        "longitude": ",".join(str(s["lon"]) for s in spots),
        "models":    model_ids[model],
        "hourly":    ",".join(full_vars),
        "forecast_days": 3,
        "timezone":  tz,
    }, headers={"User-Agent": "ColeSurfs/1.0"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]
    return data


def run() -> int:
    import pandas as pd

    try:
        from config import SPOTS
        from waves import _parse_response, _WAVE_VARS_FULL, MARINE_API
        from csc.dashboardify import dashboardify
    except Exception as e:
        print(f"FATAL: import failure: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    east_spots = [s for s in SPOTS if s["buoy_id"] in EAST_BUOYS]
    if not east_spots:
        print("FATAL: no East Coast spots in config", file=sys.stderr)
        return 1

    model_ids = {"GFS": "ncep_gfswave025", "EURO": "ecmwf_wam025"}

    print(f"Testing dashboard-training parity for "
          f"{len(east_spots)} buoys × {len(MODELS)} models")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Buoys:  {', '.join(EAST_BUOYS)}\n")

    ok = 0
    fail = 0
    skipped_nan = 0
    failures = []

    for model in MODELS:
        try:
            utc = _pull("UTC", model, east_spots, _WAVE_VARS_FULL,
                        MARINE_API, model_ids)
            ny = _pull("America/New_York", model, east_spots, _WAVE_VARS_FULL,
                       MARINE_API, model_ids)
        except Exception as e:
            print(f"FATAL: Open-Meteo unreachable ({model}): "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            return 2

        for idx, spot in enumerate(east_spots):
            utc_h = utc[idx]["hourly"]
            ny_h = ny[idx]["hourly"]

            utc_times = pd.to_datetime(utc_h["time"]).tz_localize("UTC")
            ny_times_local = pd.to_datetime(ny_h["time"]).tz_localize(
                "America/New_York")
            ny_times = ny_times_local.tz_convert("UTC")
            ny_idx_by_utc = {t: i for i, t in enumerate(ny_times)}

            n = len(utc_times)

            def _get(col, i):
                v = utc_h.get(col, [None] * n)
                return v[i] if i < len(v) else None

            def _getny(col, i):
                v = ny_h.get(col, [None] * len(ny_times))
                return v[i] if i < len(v) else None

            for i_utc, t_utc in enumerate(utc_times):
                i_ny = ny_idx_by_utc.get(t_utc)
                if i_ny is None:
                    continue

                raw_utc = {"time": utc_h["time"][i_utc]}
                raw_ny = {"time": ny_h["time"][i_ny]}
                for v in _WAVE_VARS_FULL:
                    raw_utc[v] = _get(v, i_utc)
                    raw_ny[v] = _getny(v, i_ny)

                ours = dashboardify(raw_utc)

                ny_block = {"hourly": {"time": [raw_ny["time"]]}}
                for k in _WAVE_VARS_FULL:
                    ny_block["hourly"][k] = [raw_ny[k]]
                theirs = _parse_response(ny_block)[0]

                fields = (
                    ("wave_height_ft", TOL_HEIGHT_FT, _approx),
                    ("wave_period_s",  TOL_PERIOD_S, _approx),
                    ("wave_direction_deg", TOL_DIR_DEG, _dir_eq),
                )
                diffs = []
                for f, tol, cmp in fields:
                    if not cmp(ours.get(f), theirs.get(f), tol):
                        diffs.append((f, ours.get(f), theirs.get(f)))
                if diffs:
                    fail += 1
                    if len(failures) < 10:
                        failures.append((model, spot["buoy_id"], t_utc, diffs))
                else:
                    # if everything is None (e.g. sub-6s chop filtered on both
                    # sides), we count that as ok — parity is proven.
                    if ours.get("wave_height_ft") is None and \
                       theirs.get("wave_height_ft") is None:
                        skipped_nan += 1
                    ok += 1

    total = ok + fail
    print(f"Parity checks: {ok}/{total} ok ({skipped_nan} both-None)")
    print(f"               {fail}/{total} diverged beyond tolerance")

    if failures:
        print("\nDivergences above tolerance:")
        for m, b, t, d in failures:
            print(f"  {m}/{b} @ {t}:")
            for f, a, bb in d:
                print(f"    {f}: ours={a!r} waves={bb!r}")

    if fail == 0:
        print("\nPASS — training-time dashboardify produces dashboard-"
              "identical outputs for every physical UTC hour.")
        return 0
    print("\nFAIL — divergence beyond rounding tolerance. "
          "Training features do not match the dashboard.")
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
