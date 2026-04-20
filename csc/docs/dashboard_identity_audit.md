# Dashboard-identity audit of CSC training data

## Summary

**14 divergences found, 6 high-severity.** The CSC pipeline is **structurally
wrong, major refactor required**: training features are raw Open-Meteo hourly
variables in a different quantity-space from every dashboard cell, while the
inference path (`csc.predict._rename_model_frame`) patches in
dashboard-filtered values, so the model is fit on one distribution and served
on another. Until the training ingest routes through
`csc.dashboardify._parse_response` (which already exists and is unused), every
CSC prediction is the product of a mis-specified supervised-learning problem.

The top three most severe:

1. EURO feature `euro_wave_height` is combined sea state in training but
   (attempted) primary-swell partition at inference — different quantity,
   different units, different numeric distribution.
2. `wave_peak_period` (Tp) is never written to the training archive;
   `gfs_wave_period` / `euro_wave_period` are mean period (Tm), but the
   dashboard displays Tp. Training Tp column is literally a different
   physical quantity from the displayed number.
3. Dashboard's `_build_components` filter (period ≥ 6 s, height > 0, top-2 by
   energy, fallback to combined) is never applied at training time, so model
   inputs include noise-band chop that the dashboard would have dropped
   entirely. `csc.dashboardify` exists for exactly this purpose and is
   imported nowhere outside its own `__main__` verify function.

---

## Divergences (ranked by severity)

### 1. [High] EURO `wave_height` quantity mismatch — combined vs primary-swell

- **What it is**: `gfs_wave_height` and `euro_wave_height` in the training
  wide frame are the raw Open-Meteo `wave_height` variable (combined sea
  state, m). At inference, `csc/predict.py:87-115` populates the same column
  from `waves.py`'s top-level `wave_height_ft` (primary-swell partition
  filtered to period ≥ 6 s, converted back to m). Schema says "combined"; the
  numeric values at train vs serve time can differ by 3× under wind-dominated
  conditions (ECMWF WAM wind chop).
- **How the dashboard behaves**: `waves.py::_parse_response` (lines 130-166)
  emits `wave_height_ft` from `_build_components` — highest-energy surviving
  swell partition, or combined wave_height only as a last-resort fallback
  when every partition fails the ≥ 6 s filter. Uses `wave_peak_period` as the
  first-choice period (line 135).
- **How training behaves**: `csc/logger.py:83-117` writes raw `wave_height`,
  `wave_period`, `wave_direction` directly (no filter, no fallback logic, no
  peak-period preference). `csc/data.py:_wide_forecasts` pivots these into
  `{model}_wave_height` columns fed to the model untouched.
- **Observable impact**: EURO trace inflated several-fold relative to the
  dashboard during wind chop (because ECMWF WAM wave_height includes the
  full 2D spectrum and Open-Meteo exposes no WAM partitions). This is
  exactly the "EURO looks 3× bigger than GFS" footgun the `csc.predict`
  comment at line 216 warns about.
- **Fix owner**: training-data ingest rewriter — route logger and any
  historical rebuild through `csc.dashboardify.dashboardify_series` before
  Parquet write, OR land the CMEMS WAM partition backfill so EURO has real
  `swell_wave_*` columns and remap `euro_wave_height → euro_swell_wave_height`
  throughout features/models/predict.

### 2. [High] GFS / EURO period column is Tm, dashboard displays Tp

- **What it is**: Training `gfs_wave_period` and `euro_wave_period` are the
  Open-Meteo `wave_period` variable (mean period Tm). The dashboard's primary
  display and fallback combined period is `wave_peak_period` (Tp). Comment
  at `waves.py:121-122` is explicit that Tp ≈ Tm/0.78 on ECMWF WAM, so these
  are not interchangeable.
- **How the dashboard behaves**: `waves.py:123,135` — `wp_peak` is preferred
  over `wp` for the combined fallback; partition periods come straight from
  `swell_wave_period`, which is the mean period of that partition only (fine
  for the partition, but only the partition period lands on-screen).
- **How training behaves**: `csc/schema.py:116-120` (`OM_WAVE_VARS`) does
  not include `wave_peak_period` at all. It's never requested, never
  archived, never a feature. `csc/logger.py:58` sends `om_var_columns()`
  which excludes it.
- **Observable impact**: For the combined period column every training row
  carries Tm; the dashboard cell a user sees that same hour carries Tp,
  which is ~1.28× larger on ECMWF WAM, proportionally smaller but still
  different on GFS. Learning to predict combined Tp with Tm as the input
  is a permanent bias floor.
- **Fix owner**: `csc/schema.py:OM_WAVE_VARS` — add `wave_peak_period`;
  `csc/logger.py:_long_rows_from_response` — write it; decide which of Tm
  and Tp should be the canonical `*_wave_period` column (dashboard uses Tp,
  so training should too).

### 3. [High] `_build_components` filter never applied at training time

- **What it is**: The dashboard drops partitions with period < 6 s and
  height ≤ 0, sorts remaining partitions by energy (h_ft² · p) descending,
  keeps top-2, and falls back to the combined wave only if every partition
  is filtered. Training ingests every partition unconditionally, including
  noise-band chop.
- **How the dashboard behaves**: `waves.py:_build_components` (lines 52-108).
- **How training behaves**: `csc/logger.py:_long_rows_from_response`
  (lines 73-117), `csc/schema.py:OM_WAVE_VARS` — every non-null partition
  value is emitted as a `variable,value` row.
- **Observable impact**: Features like `gfs_swell_wave_height` at training
  time include 3-4 s wind-wave energy that the dashboard would have
  discarded; the model learns a different relationship than what the
  dashboard exposes to the user.
- **Fix owner**: training-data ingest — call `csc.dashboardify.dashboardify`
  per hourly row, extract `wave_height_ft / _ft / wave_period_s /
  wave_direction_deg / components` and persist THOSE as the training
  features (in SI units for consistency).

### 4. [High] `csc.dashboardify` is dead code — never imported in pipeline

- **What it is**: `csc/dashboardify.py` is the explicit bridge from raw
  Open-Meteo → dashboard record shape. Its docstring declares it the
  "single source of truth for that conversion." Grep across the repo shows
  it is imported by exactly zero other modules (only self-import in its
  `__main__` verification routine).
- **How the dashboard behaves**: Every displayed cell is funneled through
  `waves.py::_parse_response` (which `dashboardify` wraps).
- **How training behaves**: Raw Open-Meteo variables go from
  `csc/logger.py` → Parquet → `csc/data.py::_wide_forecasts` → features →
  model. `dashboardify` is never in the loop.
- **Observable impact**: All of items #1-#3 above. The fix-shaped module
  exists; it is not wired in.
- **Fix owner**: `csc/logger.py`, `csc/experiment.py` — import
  `dashboardify_series` and apply before writing shards / before building
  the training frame.

### 5. [High] Training target (primary-swell) lacks the dashboard's 2-component cap / energy sort consistency on training path vs live serving

- **What it is**: `csc/backfill_primary_swell.py` correctly reuses
  `buoy._spectral_components` — good. But training then matches by
  `partition` ordinal (`partition=1` only, `csc/data.py:257`), while the
  live dashboard shows components ordered by energy descending. If the
  backfill ever emits in a different order than live (e.g. secondary has
  more energy than primary in one hour), training's `partition=1` is not
  the same physical row as dashboard's displayed primary.
- **How the dashboard behaves**: `buoy.py:_spectral_components` line 312
  — `components.sort(key=lambda c: c["energy"] or 0, reverse=True)`
  returns top-2 by energy.
- **How training behaves**: `csc/backfill_primary_swell.py:198` — loops
  `enumerate(comps, start=1)` to assign `partition` ordinals. Good — this
  DOES preserve the energy-descending order since `_spectral_components`
  already sorted. Verified: partition=1 IS highest-energy. **This path
  matches.** Downgrading to informational but noting the fragile coupling:
  any future change to the sort in either path breaks the contract.
- **Observable impact**: None today; high fragility. Demote to **Low**
  after verification.
- **Fix owner**: add a contract test asserting
  `_spectral_components`'s ordering is "by energy desc" and cite it from
  both `backfill_primary_swell.py` and `logger.py::log_observation`.

### 6. [High] `combined_wave_height_m` is written by `waves.py` but NOT used in training — and `predict.py` uses it at inference

- **What it is**: `waves.py:_parse_response` (lines 162-165) adds
  `combined_wave_height_m` / `_period_s` / `_direction_deg` to every
  record. `csc/predict.py:_rename_model_frame` consumes these as the v1
  training schema (`gfs_wave_height := combined_wave_height_m`). Training
  data does NOT come from `waves.py` at all — it comes from `csc.logger`'s
  raw Open-Meteo `wave_height` / `wave_period` / `wave_direction`. The
  `combined_wave_period_s` inference value prefers `wave_peak_period`
  (Tp), but `logger.py` wrote Tm — so the same `gfs_wave_period` column
  is Tm at train-time and (fallback-dependent) Tp-or-Tm at serve-time.
- **How the dashboard behaves**: Not displayed — these three keys are
  explicitly labeled in `waves.py:162` as "used by csc.predict, not the
  main UI."
- **How training behaves**: Ignored entirely.
- **Observable impact**: The exact numerical distribution of
  `{gfs,euro}_wave_period` feature at inference is different from
  training because `wave_peak_period` is substituted when present. At
  train-time Tm; at serve-time Tp most of the time, Tm on days when
  `wave_peak_period` is null. Model extrapolates.
- **Fix owner**: Either (a) remove `combined_wave_period_s`
  peak-over-mean preference from `predict.py` so inference matches
  training's Tm, or (b) add `wave_peak_period` to OM_WAVE_VARS and retrain
  on Tp — option (b) aligns training with the dashboard.

### 7. [Medium] EURO has no `swell_wave_*` columns at all in training; inference falls back to combined

- **What it is**: `OM_WAVE_VARS` includes `swell_wave_*` names, but ECMWF
  WAM via Open-Meteo returns null for all of them; observation counts in
  the training archive for any `euro_swell_wave_*` row should be ≈ 0.
  `csc/features.add_engineered` doesn't reference `euro_swell_*` (see
  `feature_columns(use_partition_features=True)` — only GFS partition
  features listed). But `csc.predict._rename_model_frame` WRITES
  `euro_swell_wave_height = euro wave_height_ft / 3.28084` (primary-swell
  from dashboard filter). So inference has
  `euro_swell_wave_height` populated with dashboard primary-swell
  height, and if `add_engineered` or some ridge variant ever adds this to
  the feature list, the serving frame will have a populated column that
  was 100% NaN in training.
- **How the dashboard behaves**: Dashboard EURO cells show the
  `_build_components` fallback (combined with Tp preference) on partition-null
  hours.
- **How training behaves**: `gfs_swell_wave_height` is populated from raw
  Open-Meteo partition (unfiltered mean-period swell height), EURO
  counterpart is empty.
- **Observable impact**: Latent schema drift — any added feature that
  references `euro_swell_*` will silently have inference distributions
  uncorrelated with zero training signal. The CMEMS backfill
  (`cmems_backfill.py`) exists to fix this but has never run (no
  `.csc_data/euro_partitions/` directory exists on disk).
- **Fix owner**: run `csc.cmems_backfill`, wire `euro_partitions/` into
  `csc/data.py::read_forecasts` or equivalent, add `euro_swell_*` features
  to `features.feature_columns`.

### 8. [Medium] Lead=0 semantics differ: Open-Meteo "analysis" vs dashboard "current"

- **What it is**: Training `lead_days=0` is Open-Meteo's analysis at the
  given valid_utc — essentially the "0 h forecast" from whatever cycle
  was active when the hour occurred. Dashboard's "current" snapshot comes
  from `fetch_all_wave_forecasts` (3600 s TTL) — whichever cycle's
  forecast was cached. The two can disagree by one cycle (6 h on GFS,
  12 h on EURO).
- **How the dashboard behaves**: `waves.py:fetch_all_wave_forecasts`
  caches a per-TTL Open-Meteo response; the hourly records span `past 0
  days` + `FORECAST_DAYS`, all tagged by the model-run cycle that was
  active at fetch time.
- **How training behaves**: `csc/logger.py:54` — `past_days=7,
  forecast_days=1`, lead_days=0 is the newest analysis captured by the
  6-hourly logger, but the valid_utc window covers 7 d back. So a
  training row at 2026-04-12 00:00 may represent the analysis captured
  on 2026-04-13 00:00 (lag 24 h) or 2026-04-12 06:00 (lag 6 h) depending
  on cycle.
- **Observable impact**: lead_days=0 is actually "latest-analysis
  for-this-hour captured within the past 24 h." Not the same as
  dashboard "current" for future hours (where dashboard uses the
  current-cycle forecast, not a future analysis).
- **Fix owner**: clarify in docs; consider whether lead_days=0 should
  only be used for historical (past) rows in training, not the rolling
  latest.

### 9. [Medium] ±30 min `merge_asof` tolerance introduces label noise vs exact-hour model

- **What it is**: `csc/data.py:build_training_frame` uses
  `pd.merge_asof(..., direction="nearest", tolerance="30min")` to match
  NDBC observations to forecast valid_utc. NDBC reports on a ~hourly
  cadence but not always on the top of the hour (41-minute ingest
  typical for realtime2). Open-Meteo hourly forecasts are exactly on the
  hour. The dashboard never merges these; it shows each independently.
- **How the dashboard behaves**: Buoy cell and model cell are separate
  rows; user's eye does the temporal merge.
- **How training behaves**: matches buoy reading at e.g. 13:41 UTC to
  forecast at 14:00 UTC (19 min apart — within tolerance). Training
  label is "buoy reading closest to the hour," not "buoy reading at the
  hour."
- **Observable impact**: Under fast-moving sea states (storm front), 30
  min of drift can be 0.3-0.5 ft in Hs. Adds irreducible noise to the
  learning target.
- **Fix owner**: either tighten tolerance to 15 min with explicit
  dropping of unmatched rows, or interpolate observations to the
  forecast grid before the join.

### 10. [Medium] Unit convention: dashboard shows feet, training uses meters — conversion path crosses ft→m→ft twice

- **What it is**: `waves.py` converts `h_m` (Open-Meteo m) → `h_ft`
  (display). Training observations are in `hs_m` (backfill, direct m)
  or `hs_ft` (live-log) normalized to `hs_m` in `data.py:85-101`.
  `predict.py:108` converts `wave_height_ft` back to meters for the
  `gfs_swell_wave_height` feature. Predictions are emitted in `pred_hs_m`
  and converted to ft in `correct_forecast` (line 221) and
  `multi_serve.py:278`.
- **How the dashboard behaves**: everything in feet.
- **How training behaves**: everything in meters.
- **Observable impact**: 3.28084 conversion factor (inverse-exact) rounds
  off at every step; typical total error from three conversion
  round-trips is < 0.01 ft — not load-bearing, but a source of spurious
  0.01-ft diffs on the comparison dashboard.
- **Fix owner**: pick one unit canonically at data layer; today SI is
  fine, conversion is lossless at float64 in practice.

### 11. [Medium] `hs_ft` → `hs_m` normalization uses 3.28084, not exact 0.3048

- **What it is**: `csc/data.py:95,100,208-210` uses `/ 3.28084` to
  convert live-log ft to m. `csc/backfill_primary_swell.py:65` uses
  `FT_PER_M = 3.28084`. `csc/predict.py:108` uses 3.28084. These are
  four-digit approximations of the exact SI conversion 1 m = 3.280839895 ft.
  `config.m_to_ft` (used by `buoy.py` and `waves.py` for display) uses
  some canonical factor — need to verify parity.
- **How the dashboard behaves**: uses `config.m_to_ft` (not sampled here).
- **How training behaves**: hardcoded 3.28084.
- **Observable impact**: Round-trip error ~8.5 ppm per conversion — 1 cm
  on a 10 ft wave. Negligible.
- **Fix owner**: centralize on one constant; reference from `config.py`.

### 12. [Medium] `wave_peak_period` written by waves.py but training never captures it

- **What it is**: See #2. Dashboard uses Tp preferentially; training has
  only Tm. The dashboard's period cell on combined-fallback hours is
  numerically different from anything in the training archive.
- **Fix owner**: `csc/schema.py`.

### 13. [Low] Engineered features are derivatives of training-raw inputs, not dashboard-matched

- **What it is**: `csc.features.add_engineered` builds `d_hs = gfs_wave_height
  - euro_wave_height`, `gfs_tp_sin_dp`, etc. These are derivatives of the
  upstream `*_wave_*` columns. Because the upstream columns are
  un-dashboardified (see #1-#3), every derivative is also un-dashboardified.
- **How the dashboard behaves**: n/a (dashboard doesn't display these).
- **How training behaves**: derives from raw-OM inputs.
- **Observable impact**: Derivatives of dashboard-matched features would be
  fine per the audit prompt; derivatives of raw-OM are all divergences. The
  fix is at the upstream inputs (#1-#3), not in features.py itself.
- **Fix owner**: no action required in `features.py` once upstream is fixed.

### 14. [Low] Observation target `partition=0` is combined WVHT, target semantics diverge from v3 README

- **What it is**: `csc/data.py:141,157` defaults
  `build_training_frame(partition=0)`, which is combined WVHT (NDBC
  realtime significant wave height of the full spectrum).
  `build_training_frame_primary(partition=1)` is primary-swell Hm0 from
  spectral decomposition. The README claims CSC predicts primary-swell;
  the v1 `build_training_frame` (used by `run_bakeoff`, which is the
  default `main()`) still targets combined WVHT. Only `run_bakeoff_primary`
  targets primary-swell.
- **How the dashboard behaves**: Dashboard buoy cell Hs = combined WVHT
  (see README README:49-51 — "CSC does NOT predict the buoy-cell top-level
  Hs"). So the v1 target actually DOES match buoy-cell Hs — but that's
  not what the dashboard GFS/EURO cells show (those are primary-swell).
- **How training behaves**: Two paths, both shipped. Which is "current"
  depends on whether `.csc_models/current/manifest.json:target` is
  `"primary_swell"` or `"combined"`.
- **Observable impact**: If the dashboard is displaying v1-combined
  predictions on panels labeled "primary-swell" (or vice versa), users
  see quantities that don't match the axis label. `multi_serve._active_target`
  handles this correctly IF the manifest is truthful.
- **Fix owner**: verify `.csc_models/current/manifest.json:target` is
  consistent with the panel labels on `/csc`; retire `build_training_frame`
  (non-primary) if primary is the only supported target going forward.

---

## Paths that DO match (sanity check)

- **Primary-swell spectral decomposition** (`buoy._spectral_components`) is
  reused verbatim by `csc/backfill_primary_swell.py` (imports at line 54)
  and by `csc/logger.py::log_observation` via `buoy.fetch_buoy` (line 603 in
  `buoy.py` hooks `log_observation` inside `fetch_buoy`). Same algorithm,
  same peak-merging thresholds (DIR_THRESH=35°, VALLEY_THRESH=0.70), same
  Hm0/Tm/circ-mean formulas, same filter (≥ 0.2 ft, ≥ 6 s). Bit-for-bit.
- **NDBC 6 s period floor** on fallback: `buoy.py:_parse_spec` line 364
  mirrors the filter in `waves.py:_build_components` line 77.
- **Direction convention**: Open-Meteo and NDBC both report "wave FROM"
  direction in degrees; no flips required. `csc/schema.py:153` calls this
  out explicitly.
- **UTC timestamps**: Both sides of the join land in
  `pd.Timestamp(..., tz=utc)` — verified in `csc/data.py:68,109,244`.
- **Buoy scope filter**: `display_filter.PUBLIC_BUOYS` matches
  `schema.EAST_BUOYS` — only the 3 East buoys surface, consistent.

---

## Recommended fix ordering

The ordering below minimizes breakage and lands the highest-value fix first.

1. **Wire `csc.dashboardify` into `csc/logger.py`** so every new forecast
   shard is dashboard-parity. One import + one helper call. Does not
   backfill — existing shards stay non-dashboardified until step 4.
2. **Add `wave_peak_period` to `OM_WAVE_VARS`** and redefine the canonical
   `*_wave_period` column as Tp. Fixes #2 and the Tp side of #1/#6.
3. **Retro-dashboardify existing `.csc_data/forecasts/` and
   `.csc_data/live_log/forecasts/` shards** — one-shot rewriter that reads
   each shard, replays `dashboardify` (can reuse `combined_wave_*` that
   waves.py already emits for post-2024-Q4 shards if `wave_peak_period`
   was captured, otherwise recompute from raw variables).
4. **Run `csc.cmems_backfill`** for EURO swell partitions; wire
   `.csc_data/euro_partitions/` into `csc/data.py::read_forecasts`; add
   `euro_swell_wave_*` features to `features.feature_columns` and adjust
   `csc.predict.correct_forecast_primary` to use `euro_swell_wave_height`
   in place of `euro_wave_height`. Fixes #1 EURO side and #7.
5. **Tighten `merge_asof` tolerance to 15 min** and re-fit; measure label
   noise reduction. Addresses #9.
6. **Replace 3.28084 hard-codes with `config.FT_PER_M`** constant.
7. Retire `build_training_frame` (combined-only) in favor of primary-only
   path; simplify `_active_target` logic; address #14.
8. Add a unit test pinning `_spectral_components` ordering to
   energy-descending so divergence #5 can't regress silently.

Steps 1-4 are the "bit-for-bit" fix; 5-8 are hygiene.
