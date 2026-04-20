# Data pipeline and train / test splits

## Sources

### Forecasts — Open-Meteo Marine API (`csc/logger.py`, `csc/backfill.py`)

Two models, called via `models=ncep_gfswave025` and
`models=ecmwf_wam025`:

- **GFS-Wave** — NOAA NCEP, 4 cycles/day (00/06/12/18Z),
  `csc/schema.py:CYCLES_PER_DAY["GFS"]`.
- **ECMWF WAM** — ECMWF, 2 cycles/day (00/12Z).

Each request asks for the hourly analysis value plus the Open-Meteo
`previous_day1..7` companions, giving lead times 24/48/.../168 h.
Variable list in `csc/schema.py:OM_WAVE_VARS` — combined sea-state
(wave_height/period/direction) and the three Open-Meteo swell partitions
(primary, secondary, wind-sea). Note: ECMWF WAM does not expose the
partitions on Open-Meteo, so `euro_swell_*` columns are always NaN.

### Observations — NDBC stdmet + CDIP (`csc/ndbc_hist.py`, `csc/backfill.py`)

- **NDBC** — NOAA standard-meteorological hourly archive, pulled from
  `ndbc.noaa.gov/view_text_file.php?filename=<id>h<year>.txt.gz`. Used
  for all East-Coast buoys and for 46025 on the West Coast.
- **CDIP** — Coastal Data Information Program, used for 46221 and
  46268. NDBC's WVHT record for these two is thin; CDIP has continuous
  coverage.
- **44258 (Halifax)** was in the original scope but the feed is
  ECCC-operated and not re-streamed to NDBC — dropped from CSC.

### Forward logger (`csc/logger.py`) — rolling seven-day capture

Each logger tick stores one hour's analysis per buoy × model AND the
six lagged forecasts (`previous_day1..7`). The archive therefore grows
with real-time valid-time coverage AND a parallel lead-time stack. v1
trains on analysis-only (`lead_days=0`); the v2 corrector will use the
full lead-time axis.

## Archive scope (current)

| source | date range | row count (approx) |
|---|---|---|
| forecasts (GFS + EURO) | 2024-07 → 2026-04 | ≈ 100k after join |
| observations (8 buoys) | 2024-07 → 2026-04 | ≈ 100k hourly |

Source: `.csc_data/forecasts/model=*/buoy=*/year=*/month=*/*.parquet`,
 `.csc_data/observations/buoy=*/year=*/*.parquet`.

Post-join training frame: **100,489 rows** after dropping rows missing
any of the core model inputs (`manifest.json:train_rows`).

## Schema — `csc.data.build_training_frame()`

Wide DataFrame, one row per (buoy × valid_utc × lead_days) after
nearest-time observation match.

| column | type | units | notes |
|---|---|---|---|
| `buoy_id` | str | — | one of 8 (see `csc/schema.py:BUOYS`) |
| `valid_utc` | datetime64[ns, UTC] | — | forecast valid time |
| `lead_days` | int | — | v1: always 0 |
| `gfs_wave_height` | float | m | analysis Hs |
| `gfs_wave_period` | float | s | analysis Tp |
| `gfs_wave_direction` | float | deg | FROM, 0-360 |
| `gfs_swell_wave_*` | float | m/s/deg | primary partition |
| `gfs_secondary_swell_wave_*` | float | m/s/deg | secondary partition |
| `euro_wave_*` | float | m/s/deg | combined analysis; no partitions |
| `obs_hs_m` | float | m | matched NDBC/CDIP WVHT (combined) |
| `obs_tp_s` | float | s | matched DPD |
| `obs_dp_deg` | float | deg | matched MWD/Dp |

**Join semantics**: `pd.merge_asof(direction="nearest", by="buoy_id",
tolerance="30min")`. Rows where no observation matched within ±30 min
are dropped.

## Train / test split

### Decision: 4-fold month-based CV (`csc/cv.py`)

**Why we moved off the single 80/20 time holdout:** the old split
placed the 20 % test window entirely in fall/winter. Seasonal coverage
of that test set was approximately 0 % summer and 0.1 % spring —
meaning any reported gain vs a baseline was only known to hold during
fall/winter regimes. That's a real overfitting-to-season risk. The
4-fold, season-balanced CV restores a fair picture of generalization.

### Fold composition (current artifact)

From `.csc_models/current/folds.json` / `manifest.json:cv`:

| fold | train rows | test rows | test months | DJF / MAM / JJA / SON count |
|---|---:|---:|---|---|
| 0 | 66,801 | 33,688 | 2024-07, 09, 12; 2025-03, 08, 10; 2026-01 | 2 / 1 / 2 / 2 |
| 1 | 73,745 | 26,744 | 2024-08, 10; 2025-01, 04, 11 | 1 / 1 / 1 / 2 |
| 2 | 78,015 | 22,474 | 2024-11; 2025-02, 05, 06 | 1 / 1 / 1 / 1 |
| 3 | 82,906 | 17,583 | 2025-07, 09, 12; 2026-04 | 1 / 1 / 1 / 1 |

Algorithm (`cv.py:_assign_fold_per_month`): round-robin each season's
sorted month list across the 4 bins. Fold i's test = union of bin-i
across all four seasons; train = complement.

Properties:

- **Every fold touches every season.** No fold is purely fall/winter.
- **Deterministic.** Seed is implicit in the sorted (year, month) list.
  Re-running on the same archive reproduces the same split.
- **Per-coast evaluation respected.** `lgbm_east` and `lgbm_west` are
  scored only on matching-coast rows in each fold, recorded as
  `scope="east"/"west"` in `metrics.json`.

### Buoy scope and coast mapping

| coast | buoy | label | operator |
|---|---|---|---|
| east | 44013 | Boston (MA) | NDBC |
| east | 44065 | NY Harbor Entrance | NDBC |
| east | 44091 | Barnegat (NJ) | NDBC |
| east | 44097 | Block Island Sound | NDBC |
| east | 44098 | Jeffrey's Ledge (NH) | NDBC |
| west | 46025 | Santa Monica Basin | NDBC |
| west | 46221 | Santa Monica Bay | CDIP |
| west | 46268 | Topanga Nearshore | CDIP |

Coast mapping lives in `csc/per_coast.py:EAST_BUOYS`, `WEST_BUOYS`,
and `coast_of()`.

## Known limitations

- **Training target is combined WVHT, not spectral primary swell.**
  NDBC's WVHT is significant-wave-height of the full spectrum; the
  dashboard surfaces primary-swell Hs from `buoy.py`'s spectral
  decomposition. CSC v1 is therefore calibrated against the combined
  sea; predictions are still shown on primary-swell panels via
  `predict.py`'s lookup shim. v2 should re-target onto spectral
  Hm0(primary).
- **Lead-time dimension is collapsed.** v1 applies one analysis-level
  correction at every lead. The forward logger is already writing the
  multi-lead archive; v2 will consume it.
- **Persistence and raw_gfs are identical on the current hold-out
  because the persistence baseline has no observation series available
  at inference time** and falls back to raw GFS. See `models.py:
  PersistenceBaseline.predict`.
