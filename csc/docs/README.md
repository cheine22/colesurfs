# CSC — Colesurfs Correction (v3, East Coast)

Statistical post-processing layer that consumes GFS-Wave and ECMWF-WAM wave
model output (served via Open-Meteo), plus engineered and seasonal features,
and emits a single bias-corrected primary-swell forecast — **the same
quantity the main `/` dashboard displays in its GFS and EURO cells**.

CSC trains against NDBC / CDIP buoy observations decomposed into swell
partitions by `buoy._spectral_components` (the exact same decomposition the
main dashboard runs live to populate its "swell components" row). This means
CSC's output is directly comparable to the GFS and EURO cells a user already
sees at `/`.

## v3 scope (East Coast)

Training buoys — each with ≥ 2 years of spectral-partition observations:

| buoy | label | spectral archive | earliest |
|---|---|---|---|
| 44013 | Boston (MA) | NDBC THREDDS + realtime | 2017 |
| 44065 | NY Harbor Entrance | NDBC THREDDS + realtime | 2017 |
| 44097 | Block Island Sound | NDBC THREDDS + realtime | 2017 |

Deferred buoys — deployed mid-2025, will auto-promote to East scope when
their archives pass 24 months (tracked in `csc/schema.py::FUTURE_EAST_BUOYS`):

| buoy | label | deployed | eligible |
|---|---|---|---|
| 44091 | Barnegat (NJ) | 2025-06-25 | 2027-06-25 |
| 44098 | Jeffrey's Ledge (NH) | 2025-08-19 | 2027-08-19 |

West Coast (46025 Santa Monica Basin, 46221 Santa Monica Bay, 46268
Topanga Nearshore) **runs silently in parallel** on a separate launchd
job (`com.colesurfs.csc-train-west`) with distinct artifact storage
(`.csc_models_west/`). West artifacts never surface on the main
dashboard or `/csc`.

## What CSC predicts vs what the dashboard shows

**Primary-swell Hs/Tp/Dp** — defined as the highest-energy coherent swell
partition filtered to period ≥ 6 s. This matches:

- The main dashboard's GFS/EURO cells (`record.wave_height_ft` from
  `waves.py::_parse_response`).
- The buoy's "swell components" row on the main dashboard (from
  `buoy._spectral_components`).

**CSC does NOT predict the buoy-cell top-level "Hs"** on the main
dashboard — that number is NDBC combined WVHT (total sea including wind
chop), which is a pre-existing asymmetry in colesurfs, not something
CSC introduced. CSC is aligned to the model cells, not the buoy cell.

## Data lineage

```
  Open-Meteo Marine API          NDBC THREDDS NetCDF
  (GFS, EURO forecasts)          (spectral E(f) + α1)
         │                              │
         │                              ▼
         │                   csc.backfill_primary_swell
         │                   (also: CDIP for West, realtime for all)
         │                              │
         │                              ▼
         │                   buoy._spectral_components
         │                   (same algorithm as live dashboard)
         │                              │
         ▼                              ▼
  csc.logger.log_forecast      .csc_data/primary_swell/
         │                              │
         ▼                              │
  .csc_data/forecasts/                  │
  .csc_data/live_log/                   │
         │                              │
         ▼                              │
  csc.data.build_training_frame_primary ◄──┘
         │
         ▼
  csc.features.add_engineered
         │
         ▼
  csc.experiment.run_bakeoff_primary(scope='east')
         │
         ▼
  .csc_models/<YYYY-MM-DD_HHMMSS>_east_v3/
         │
         ▼ (user-approved promote)
  .csc_models/current  ──►  csc.multi_serve.fetch_all_variants_live
         │                              │
         ▼                              ▼
  reads manifest.target        Flask /api/csc/variants/<buoy_id>
                                        │
                                        ▼
                                  /csc dashboard
```

## Directory map (`csc/`)

### Core pipeline
- `schema.py` — buoy list, scope definitions (`EAST_BUOYS`, `WEST_BUOYS`, `FUTURE_EAST_BUOYS`), archive paths, scope helpers (`buoys_for_scope`, `models_dir_for_scope`).
- `data.py` — Parquet readers + `build_training_frame_primary()` (v3 primary-swell target).
- `features.py` — `add_engineered()`: disagreement features, interaction terms, sin/cos direction, DOY seasonality, period-band categorical.
- `models.py` — shared fit/predict/save/load API; baselines (raw_gfs, raw_euro, mean, persistence, ridge_mos) and global LightGBM variants.
- `funplus.py` — LightGBM with sample-weighted loss favoring FUN+ conditions + FUN+ eval CLI.
- `per_coast.py` — east/west specialist LightGBMs; cross-coast router excluded from scope-filtered runs.
- `cv.py` — `kfold_month_split` (season-balanced) + `stratified_kfold_month_split` (adds Hs × period × direction stratification).
- `experiment.py` — scope-aware bakeoff: `run_bakeoff_primary(scope='east'|'west')`.
- `train.py` — `python -m csc.train --primary --scope east|west` wrapper.
- `promote.py` — atomic `.csc_models/current` flip.

### Data acquisition
- `backfill_primary_swell.py` — historical pull from NDBC THREDDS NetCDF (primary), NDBC realtime `.data_spec`/`.swdir` (45-day rolling), CDIP THREDDS (West only).
- `ndbc_hist.py` — NDBC stdmet historical helper (legacy path; kept for completeness).
- `logger.py` — forward logger: per-6h pull of each model's `previous_day1..7` + live-fetch hook persists buoy observations.

### Serving
- `predict.py` — inference entry; loads from `.csc_models/current`.
- `serve.py` — TTL-cached wrapper; Flask routes consume this.
- `multi_serve.py` — multi-variant compare endpoint helper; reads `manifest.target` to pick primary-swell vs combined display.
- `dashboard_eval.py` — `/csc` panel data producer.
- `display_filter.py` — public scope filter (East buoys only on `/csc`).

### Evaluation
- `evaluate.py` — general verification-metric library (bias, MAE, RMSE, SI, HH, r, ACC, NSE, POD/FAR/CSI, circular MAE, vector r).
- `surf_metrics.py` — surfer-relevant leaderboard: daytime-filtered FN/FP counts for FUN and SOLID+ categories, period MAE, FUN+ Hs MAE, composite ranking.
- `continuous_eval.py` — weekly cron-driven rolling-30-day re-score of every preserved artifact.
- `notify.py` — Pushover-first notification on seasonal retrain completion.

### UI support
- `glossary.py` — term definitions that back the `/csc` glossary panel.

## Running

```bash
# One-time historical backfill (reruns are idempotent per year)
python -m csc.backfill_primary_swell

# Bakeoff + train
python -m csc.train --primary --scope east     # East Coast (dashboard-facing)
python -m csc.train --primary --scope west     # West Coast (silent)

# Promote the newest East artifact
python -m csc.promote .csc_models/<YYYY-MM-DD_HHMMSS>_east_v3

# Surfer-relevant leaderboard
python -m csc.surf_metrics --scope east
```

Automation (launchd):

| plist | schedule | purpose |
|---|---|---|
| `com.colesurfs.csc-log` | every 6 h | forward logger — builds the lead-time archive |
| `com.colesurfs.csc-train` | Mar 21 / Jun 21 / Sep 21 / Dec 21 @ 03:15 | seasonal East Coast retrain + Pushover notification |
| `com.colesurfs.csc-train-west` | same dates @ 03:30 | silent West Coast retrain |
| `com.colesurfs.csc-eval` | every Monday @ 06:00 | weekly continuous eval |

## How a new engineer should read this

1. `README.md` (you're here) — 5 min orientation.
2. `strategy.md` — why each non-obvious decision was made.
3. `data-and-splits.md` — when you need to trust the numbers.
4. `metrics.md` / `models.md` — reference.
5. `model-comparison.md` + `top-3-rationale.md` — which models are currently best on what surfers care about, and why.
