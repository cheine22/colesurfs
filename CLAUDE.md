# CLAUDE.md — colesurfs

Guidance for Claude when editing this repo.

## What this is

Single-page Flask app that aggregates surf-forecast data (NOAA NDBC buoys,
NOAA CO-OPS tides, Copernicus Marine / Open-Meteo wave + wind models) and
renders it as a swell table. No build step and no bundler —
`templates/index.html` inlines all JS and CSS in one file.

Alongside the main dashboard, the `csc2/` package is a forecast-correction
model. It trains on paired (EURO forecast, GFS forecast, buoy observation)
triples to predict corrected primary+secondary swells. Trained models live
in `.csc2_models/`; the top performer surfaces on the main dashboard via
the CSC2 (beta) toggle, and `/csc` is the eval page (archive table, model
defs, metric tables).

## Where things live

- `app.py` — Flask routes, `Cache-Control` rules, background cache warmer,
  `/api/buoy_historical_context` (per-hour observed + model-agreement
  record, backed by `.csc2_data/forecasts/` reads for CSC2 buoys)
- `buoy.py` — NDBC fetch + spectral swell decomposition. `fetch_buoy_history`
  defaults to a 10-day range; each record carries a raw
  `spectrum: [[freq_hz, energy_density_m2/Hz, direction_deg | null], …]`
  field, sourced from the same `.data_spec` + `.swdir` bytes already parsed
  for component decomposition (no extra HTTP)
- `waves.py` — Open-Meteo GFS-Wave partition fetch (EURO lives in CMEMS)
- `waves_cmems.py` — Copernicus Marine ANFC EURO fetch + shared processing
  pipeline (Tm01×1.20, 5 s filter, energy-sorted top-2)
- `wave_common.py` — shared `_safe`/component-builder/record-schema used by
  both wave modules. Behavior locked by `development-assets/tests/test_wave_identity.py`
  (golden fixtures); regenerate goldens only for intentional changes via
  `development-assets/tests/regen_golden.py`
- `wind.py`, `tide.py`, `sun.py` — other data sources. `fetch_all_spot_winds`
  batches all spot current-winds into one Open-Meteo call; per-spot
  `fetch_spot_wind` remains as fallback
- `cache.py` — TTL cache + disk write-through + API-call counter. Per-key
  single-flight locks so concurrent misses can't stampede a slow upstream
  (CMEMS cold fetch ≈ 90 s)
- `config.py` — loads `regions.yaml`; defines palette, wind bands, grid
- `regions.yaml` — single source of truth for regions / buoys / spots
- `swell_rules.py` + `swell-categorization-scheme.toml` — swell → color
- `templates/index.html` — the main dashboard frontend, ~5 k lines
  (see "index.html landmarks" below)
- `templates/csc.html` — the CSC2 eval page
- `gland.py` + `templates/gland.html` — the `/gland` page (Grajagan, East
  Java). Deliberately NOT wired through `regions.yaml`: there is no NDBC
  buoy and no CO-OPS station within thousands of km of G-Land, so the
  dashboard's whole data spine is inapplicable. Its own sources are
  GFS-Wave via Open-Meteo Marine, ECMWF-WAM via `waves_cmems.fetch_cmems_point`
  (already lat/lon-generic), Open-Meteo `sea_level_height_msl` for tide,
  Open-Meteo for wind, and AODN/IMOS near-real-time wave buoys off Western
  Australia as upstream sentinels. See "gland.py landmarks" below.
- `gland-swell-categorization.toml` + `templates/gland-tuner.html` — G-Land's
  OWN FLAT..MONSTRO thresholds and the `/gland/tuner` page that edits them.
  Deliberately a separate file from `swell-categorization-scheme.toml`:
  the site-wide scheme is tuned for NY/NJ beach breaks and G-Land is a
  long-period Indian Ocean point break. `gland.load_gland_bands()` /
  `categorize_gland()` keep their own cache and never call
  `swell_rules.load_bands()`, so editing one scheme cannot move the other —
  verified by a 924-cell height×period sweep. Only the category *names* and
  *colours* are shared. `/gland/tuner` and `/api/gland/tuner/save` are behind
  the same LAN-only `_restrict_tuner` gate as `/tuner`.
- `development-assets/docs/gland-cheatsheet.md` — the researched forecasting
  notes behind the page's cheat-sheet panel, with sources
- `csc2/` — CSC2 package (see below)
- `favicon.svg` + `favicon-{16,32,192}.png` + `apple-touch-icon*.png` — the
  liquid-glass icon set (2026-07). The glass layers (edge refraction with
  chromatic fringe, convex sheen, lip highlight + tube caustic, foam frost,
  rim light, depth grade) are composited as pure SVG over the wave photo
  embedded in `favicon.svg`; the PNGs are re-exported from that SVG (32 px
  drops the fringe/spray, 16 px keeps only grade + lip + sheen;
  apple-touch is square full-bleed since iOS applies its own mask).
  Prior icon sets are archived in `development-assets/old-icons/`.
- `interface-guide.png` — annotated interface guide, kept at repo root.
  Living documentation: update it alongside README/CLAUDE.md whenever
  visible UI changes land (see "Interface guide" landmark below)
- `development-assets/` — dev-only material, gitignored and never synced
  to GitHub: golden-fixture test suite (`tests/`), design mockups
  (`design-demo/`), doc sources (`docs/`, incl.
  `make_interface_guide.py`), icon archive (`old-icons/`), local-dev
  launcher (`_local-launch.command`)
- `_hold/` — staging for files awaiting manual review/deletion (gitignored)
- Deployment specifics (how the app is served, restarted, tunneled) live in
  a local-only `hosting.md` that is intentionally git-ignored. Check the
  working directory for it when deploy-related questions come up.

## csc2/ package

- `csc2/schema.py` — buoy scope (5 east + 3 west), path layout, forecast-row
  columns. Every csc2 module imports `BUOYS` / paths from here
- `csc2/logger.py` — live forecast logger (`com.colesurfs.csc2-logger`,
  3 AM + 3 PM ET). Pulls CMEMS + GFS via `waves_cmems.fetch_cmems_point` /
  `waves.fetch_wave_forecast` and writes per-cycle parquet shards.
  (Label renamed from `csc2-log` 2026-07-05: the old label's launchd state
  became unspawnable — persistent EX_CONFIG even after re-bootstrap — while
  identical plist content ran fine under a new label. Old plist kept as
  `.disabled-poisoned-label`.)
- `csc2/obs_logger.py` — live NDBC observation logger
  (`com.colesurfs.csc2-obs`, every 30 min). Appends to the shared
  `.csc_data/live_log/observations/` tree with dedup on (valid_utc, partition)
- `csc2/train.py` — trainer for both architectures. Asserts the time split
  (max train cycle < min test cycle) and records per-target row counts +
  the inclusion rule in meta.json
- `csc2/predict.py` — inference (`predict_for_cycle`)
- `csc2/registry.py` — model discovery + `select_top3` ranking. The #1 slot
  additionally requires sw1_height skill ≥ 0 (`SW1_HEIGHT_SKILL_FLOOR`) —
  a high composite can't mask a model that's worse than raw EURO on
  primary height
- `csc2/eval_live.py` — daily live-eval pass (`com.colesurfs.csc2-eval`,
  5 AM ET). For every model under `.csc2_models/east/`, re-runs inference
  on cycles that post-date its training run, compares against obs that have
  since landed, and appends one row per (model, eval_date) to
  `.csc2_data/live_eval/<model_name>.parquet`. The registry's composite
  skill stays training-holdout-based; live-eval is informational.
  Not yet built: surfacing live skill on `/csc`, and a drift watchdog
  (flag when rolling-30d skill drops ~25 % below training-holdout skill)
- `csc2/gee_backfill.py` — historical EURO backfill via Google Earth Engine
  ImageCollection (`COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H`).
  Cycle-preserving archive back to 2025-04-28
- `csc2/aws_gfs_backfill.py` — historical GFS backfill via AWS S3
  (`noaa-gfs-bdp-pds`) with byte-range GRIB2 fetches driven by `.idx` sidecars
- `csc2/ndbc_backfill.py` — historical buoy-obs backfill from NDBC stdmet
  yearly archives (partition=0 / combined sea only)
- `csc2/ndbc_spectral_backfill.py` — historical buoy spectral decomposition
  (partition=1 / partition=2). Reuses dashboard-identical
  `_spectral_components` from `buoy.py`. Three sources, in fallback order:
  yearly closed (`data/historical/swden,swdir/`), monthly closed
  (`data/swden,swdir/<Mon>/`), realtime (~45 days). Buoys 44091/44097/44098
  are not NDBC-archived (USACE/UCONN/UNH-owned), so realtime is their only
  source — invoke `--realtime` to cover them. Output:
  `.csc_data/observations/buoy=<id>/year=Y/spectral[-YYYY-MM | -realtime].parquet`
  with the same schema as stdmet, populated for partition=1/2 only
- `csc2/cdip_spectral_backfill.py` — CDIP-sourced spectral backfill for
  west-coast buoys
- `csc2/archive_status.py` — computes paired-cycle coverage per buoy with
  file-cache; backs `/api/csc2/archive_status`

Local-only data directories (gitignored):
- `.csc_data/observations/`, `.csc_data/live_log/observations/` — buoy obs
- `.csc2_data/forecasts/model={EURO,GFS}/buoy=<id>/year=Y/month=M/cycle=*.parquet` — forecast shards
- `.csc2_data/live_eval/<model_name>.parquet` — daily live-eval rows
- `.csc2_data/archive_status_cache.json` — cached `/api/csc2/archive_status` payload
- `.csc2_models/east/`, `.csc2_models/west/` — trained model weights

Retrain cadence: quarterly via `com.colesurfs.csc2-retrain` (1st of
Mar/Jun/Sep/Dec at 04:00 local). Wipes the archive-status cache, recomputes
coverage, then runs `python -m csc2.train --version v1 --force` for both
architectures; auto-derives YYMMDD + coverage so naming stays correct.
Consider an off-cycle retrain when: the top performer's live skill drops
≥25 % for 3+ consecutive days, east-pool paired-cycle coverage gains
≥30 days since last train, or a new buoy-data source is backfilled (new
`spectral-*.parquet` under `.csc_data/observations/`).

### Nomenclature

- **CSC2** refers to the *training dataset* (full GFS + EURO model runs paired
  with buoy obs), not any model. A "CSC2 model" is always a trained instance
  with a name following the convention below.
- **Model instance name:** `CSC2+{baseline|ML}_{YYMMDD}_{coverage}_v{N}`
  - `baseline` vs `ML` — architecture per README §CSC2 ("CSC2 baseline" =
    per-[buoy × lead-hour × variable] linear bias correction; "CSC2 ML" =
    LightGBM GBT over EURO/GFS/delta features + lead hour + DOY). The
    baseline supports count-weighted lead-hour bias smoothing
    (`--lead-smoothing`, default ±2 h; 0 disables).
  - `YYMMDD` — train date in UTC, sorts lexicographically (e.g. `260424`).
  - `coverage` — fraction of 365 (always 365, not 366) where the **east-coast
    pool has ≥1 paired GFS + EURO + spectral-swell-buoy day**, rounded to
    0.01. "Spectral-swell-buoy" means partition=1 (primary swell) or
    partition=2 (secondary swell) from the dashboard's spectral
    decomposition (`buoy._spectral_components`); partition=0 (combined
    sea, basic NDBC stdmet) is **not** trainable because it doesn't match
    the dashboard quantity the model is predicting against. Computed as
    `len(histograms.combined_east.paired_by_doy) / 365` from
    `archive_status_cache.json` (with `BUOY_OBS_PARTITIONS = (1, 2)` in
    `archive_status.py`). The metric counts unique paired calendar dates
    uncollapsed across years, so once we cross into year 2 the value can
    exceed 1.0.
  - `v{N}` — architecture/hyperparameter variant trained on the same date's
    snapshot. Bump for any structural change (different feature set,
    different LightGBM params, different baseline binning, etc.).
- **Examples:** `CSC2+baseline_260424_0.77_v2`, `CSC2+ML_260424_0.77_v2`.
- **Weights land in:** `.csc2_models/east/<full-name>/`. The west track uses
  the identical convention under `.csc2_models/west/<full-name>/` and never
  surfaces on the dashboard until explicitly promoted.

### GFS combined-sea fallback

GFS drops swell partitions beyond ~5 days. The dashboard's
`waves.py:_parse_response` synthesizes a primary swell from combined
Hs/Tp_peak/Dp when partitions are absent. The forecast logger writes raw
partition data (sw1=null when partitions absent) PLUS the combined_*
columns alongside, so the dashboard quantity can always be reconstituted
from disk.

The CSC2 trainer mirrors this fallback at read time in
`csc2.train._apply_dashboard_fallback_gfs`: when `gfs_sw1_height_ft` is
null and `gfs_combined_height_m` is populated, sw1 is filled from the
combined fields (m→ft for height) and tagged with
`gfs_sw1_source = "combined_fallback"`. Rows where both are null are
tagged "missing" and excluded from training. This keeps on-disk shards
raw (preserving the partition-vs-fallback distinction) while making
training inputs byte-identical to dashboard rendering.

EURO has no equivalent fallback (honest-empty policy: CMEMS partition-null
cells are genuinely empty, not fallback-eligible).

## gland.py landmarks

- **Swell is sampled offshore, not at the spot** (`SWELL_NODE_LAT/LON`).
  The Blambangan peninsula shadows the inshore model cells: GFS-Wave at the
  cell the pin snaps to (-8.75, 114.25) read 11.3 s from 180° while the next
  cell south (-9.00, 114.25) read 15.3 s from 209° in the same hour — the
  inshore cell has lost the long-period SSW entirely. Both models now read
  -9.00, 114.25 (~32 km SSW), which matches Surfline's deepwater convention
  and keeps GFS-vs-EURO like-for-like. **Wind and tide still come from the
  spot itself** — only waves move offshore. Do not "fix" the node back to
  the pin.
- **Geography is traced, not invented.** `SECTIONS[*].lat/lon`, `REEF_LINE`,
  `HARBOUR_CHANNEL` and `POINT_TIP` were read off Esri World Imagery
  (2026-07) by pixel→latlng conversion, validated by the trace landing on
  Surfline's pin to 4 decimal places. The point tip is at the **SW**; the
  reef runs **north-east** into Grajagan Bay, so sections order
  Kongs → Moneytrees → Speed Reef → (harbour channel) → Chickens →
  Tiger Tracks with longitude increasing. An earlier schematic had this
  backwards.
- **`SWELL_BANDS` overlap on purpose.** 165-190 outer / 190-210 Speedies /
  **205**-250 Moneytrees / 250-285 outer — so 205-210° feeds Speedies *and*
  Moneytrees. Consumers must handle multiple matches (`bandsFor()` returns a
  list, not a hit); the dial splits overlapping bands radially via the `ring`
  field so both light up. Keep in sync with `SECTIONS[*].best_dir` — the dial
  legend and the scoring must not disagree.
- **The forecast table prints exact degrees, not compass points** — a
  deliberate departure from the main dashboard's `toCard()`. At G-Land the
  section a swell feeds turns on a few degrees (the 205-210° overlap), and a
  22.5°-wide compass point cannot resolve that. Don't "harmonise" it back.
- **The point map carries no live data.** Section markers and cards show
  *preferred* size/direction/period/tide only. Live rating and wind belong
  in the forecast table, not on the map.
- **Swell-window filtering (`pick_gland_swell`)** — the reason this page
  exists as its own module. The dashboard's energy-sorted "primary swell"
  is *wrong* at G-Land: in the dry season the largest partition is
  routinely a 7-8 s SE trade windsea that the west-facing point never
  sees, while the surf is a smaller 16 s SSW line. Partitions are scored
  as `H²·T · window_fit`, with a hard 9 s period floor
  (`MIN_GROUNDSWELL_PERIOD`) and a direction taper across
  `WINDOW_EDGE` (165-285°) peaking over `WINDOW_CORE` (200-240°, per
  Surfline). Both models are ranked the same way so model-vs-model stays
  apples-to-apples.
- **`rank_sections`** — G-Land is five waves, not one, so the page ranks
  the reef rather than rating the spot. Score is
  `(dir 30 + period ≤15 + tide 30 + wind 25) × size_fit × ceiling × prestige`.
  Note **size is a multiplicative gate, not an additive term** — Speedies
  on a 3 ft day must be 0, not "a bit off"; an earlier additive version
  gave it 27/100 off good wind and tide alone. `_quality_ceiling` caps the
  whole reef by absolute swell size so a relative winner in marginal surf
  still reads marginal. `prestige` encodes that Moneytrees/Speed Reef are
  world-class and Chickens is, per Surfline, "a slightly lame little left".
- **Per-section wind** — there is no single shore-normal here. The tip
  faces due west and the shoreline swings to NNW as the wave wraps, so
  each section carries its own `offshore` bearing (Kongs wants E, inside
  wants SE; ESE is best overall). `wind_for_section` rates against that.
- **Section scoring is server-side only.** `_build_timeline` merges all
  sources onto one hourly timeline and emits a compact `sec_gfs` /
  `sec_euro` score array per hour in `SECTIONS` order. `gland.html`
  renders those numbers and never recomputes them — an earlier draft
  duplicated the scoring in JS and the two drifted immediately.
- **Translation layer (`translate_upstream`)** — turns WA buoy readings into a
  G-Land arrival, following Collard/Ardhuin/Chapron (2009) swell tracking:
  back-project each buoy along its great circle at `cg = gT/4π`, triangulate
  the single storm position that explains *every* buoy's direction **and**
  agrees on when it radiated (`_fit_source`, coarse 2° grid then 0.5° refine),
  then forward-project that source to G-Land. Height uses geometric spreading
  only (`_spread_factor`, energy ∝ 1/[α·sin α]); dissipation is deliberately
  unmodelled rather than fudged. **The first implementation picked each buoy's
  source independently by a storm-belt heuristic and averaged — sources
  scattered by 3,700 km and it was meaningless. Confidence must come from rays
  actually converging** (`bearing_err_deg`, `time_spread_h`), not from how many
  buoys happen to share a period bin; a lone buoy always fits perfectly and
  means nothing.
  **Unvalidated.** There is no backtest — no archive of past buoy readings
  paired with what G-Land actually did. The only check run so far is a
  single-instant comparison against GFS/EURO (2026-07-29): height landed within
  0.2 ft and period carried through, but **arrival direction was 18-23° off the
  models**, which is wider than the 20°-wide Speedies band — so it cannot be
  trusted to say which section a swell will feed. Sensitivity is quantified:
  500 km of source-position error moves arrival bearing only 6.5° but shifts
  ETA by 13 h, so ETA is the fragile output. Treat the whole layer as a
  cross-check on the models, never as a replacement, and don't let the UI
  wording drift toward claiming more.
- **`compare_translation_to_models`** puts that cross-check on the page: for
  each in-window cluster it looks up the timeline row at the hour the swell is
  predicted to land (arrive_epoch is UTC; timeline keys are WIB = UTC+7 — the
  +7 h shift and the round-to-nearest-hour are easy to get wrong) and reports
  signed ΔHs / ΔT / ΔDir against GFS and EURO with an agree/close/diverge
  verdict. Per-cell colours are per-axis while the verdict combines them, so a
  green ΔHs next to a red verdict is correct, not a bug — it means height
  agrees and direction doesn't, which is the method's known failure mode. This
  comparison is legitimate rather than circular **because the translation has
  no wave model anywhere in its chain** — it is in-situ readings plus
  geometry. If anyone ever seeds the translation from model data, this panel
  stops meaning anything.
- **`fetch_upstream_model_swell` uses COMBINED sea state on purpose.** A
  waverider reports total Hs/Tp/Dp, so combined is the like-for-like against an
  observation. This is the one place on the page that touches combined values,
  and it is model-vs-buoy; the G-Land forecast table stays on primary swell,
  where model-vs-model belongs. Don't "unify" the two.
- **Upstream buoys are sentinels, not intercepts.** `UPSTREAM_BUOYS` are
  2,300-2,900 km away down the WA coast and are NOT on G-Land's swell
  rays; the page says so explicitly. `north_offset` is how far off due
  north G-Land sits from each buoy, and `transit_hours` is a distance
  scale reference, not an ETA. Don't let either get relabelled into a
  promise about arrival.

## index.html landmarks

- **Fun+ Days column** — `computeModelOverview(spotName)` (samples both
  models on a fixed 3-hour stride regardless of UI resolution; tracks
  best `min(GFS_cat, EURO_cat)` for the cell colour and counts days
  with ≥2 daytime windows ≥ FUN). A window additionally passes a
  region-wind gate: ≥1 spot in the buoy's region must rate
  Textured-or-better (per `regionWindData` + `windCondition`) at that
  hour; the gate is skipped ("honest-empty") while region wind hasn't
  loaded. Denominator = span of sampled future times in days. Cell
  rendered between `spot-cell` and `buoy-col` with class
  `model-overview`.
- **Interface guide** — `interface-guide.png`, an annotated
  production screenshot (numbered features + legend). Regenerate after
  visible UI changes: headless-Chrome capture of `:5151` at 1440×1026
  `--force-device-scale-factor=2`, composed by
  `development-assets/docs/make_interface_guide.py` (badge coords are
  layout-specific; rebuild them from the fresh capture). The image lives
  at repo root and is maintained like README — regenerate it whenever a
  visible UI change lands.
- **Region clean-wind** — `_regionCleanWind(region, data=regionWindData)`
  computes, per region per hour, whether ≥1 spot rates Glassy/Groomed/Clean
  for whichever wind model's data is passed. `_windHatchState(region, t, data)`
  returns tri-state `'solid'` (≥1 clean) / `'hatched'` (known, none clean) /
  `null` (no wind record — fetch gap or hour outside the wind window). Drives
  the white wind agreement chip (see Agreement chips), evaluated against BOTH
  `regionWindData` (active) and `regionWindAlt` (hidden model). `_cleanWindCache`
  is a WeakMap keyed on the wind-data object, so each model's data caches
  independently and model switches / refreshes / snapshot loads invalidate
  automatically. `setShowHistory` re-fetches both models' region_wind with
  `past_days` in ALL modes (not just Regional) so historical hours are covered.
  NB the state name `hatched` is vestigial — there is no visual hatch overlay
  anymore (removed 2026-07, see Agreement chips history).
- **Agreement chips** — `_agreementChips(spotName, t)` renders up to two small
  tinted letter chips (`.agreement-chip`) stacked vertically in a forecast
  cell's top-right corner (`.agreement-chips`, a right-aligned flex column;
  only from `buildWaveCell`, including the "—" below-threshold path — not on
  buoy-now or historical cells). Each chip is its letter in colour `col` on a
  `color-mix(in srgb, col 13%, transparent)` wash.
  **Swell agreement chip ("M", `.swell`):** a SOLID block in the HIDDEN
  (non-active) model's category colour (`_swellAgreementColors` returns
  `[fill, ink]` = that category's text colour + cell background, so the letter is
  knocked out of the fill) — matches the cell's own colour when the models agree,
  reveals the other model's rating when they diverge (a peek without a model
  switch). Shown ONLY for poor-or-better hidden reads (WEAK+); a FLAT or
  below-threshold hidden read, or a missing alt record, → no swell chip. The W
  chip keeps the 13% wash; both share the same geometry so the stack stays
  aligned. (v1.11.1 briefly tried a `min(active, hidden) ≥ FUN` gate to suppress
  WEAK/FUN splits — reverted before release: it left ~2 chips per 1 600
  spot-hours. The visibility problem it was aimed at was a *contrast* problem,
  solved by the solid fill; variants weighed in
  `development-assets/design-demo/agreement-chip.html`.)
  **Wind agreement chip ("W", `.wind`):** `col` is neutral-white (`var(--text0)`,
  so white in dark mode, near-black in light), shown ONLY when BOTH models
  report ≥1 clean spot in the region at that hour (`_windHatchState` === `'solid'`
  for both `regionWindData` AND `regionWindAlt`) — a cross-model clean-wind
  agreement. If either model is missing data or reads not-clean → no wind chip.
  Readings are vertically centered (`.cell-inner` justify-content:center).
  The info modal legend (`_populateAboutLegends`) renders the same M/W chips.
  **No-overlap:** forecast cells carry a `wave-cell` class; `td.cell.wave-cell
  .cell-inner` gets a right-gutter (`padding-right` 13px desktop / 12px mobile)
  so the swell reading never runs under the chips, and the readings dropped the
  `&nbsp;` separators (spacing now via `.comp-line { gap: 3px }`) to reclaim that
  gutter width — net result the columns got ~3px NARROWER, not wider.
  History: replaced a full-width `.model-agree` text pill (landed on ~88% of
  cells, buried the reading) → filled colour dots → these M/W chips. The wind
  rule also moved twice — cross-model hatch *concordance* (both agree either
  way) → active-model clean-wind only (no alt fetch) → cross-model clean-wind
  *agreement* (both must be clean), which is why the alt-model region-wind
  fetch (`regionWindAlt`) is loaded alongside `regionWindData` everywhere the
  latter is (re)loaded.
- **Historical strip** — `_buildHistoricalCellsHtml(stationId, resolutionHours)`
  + `buildHistoricalCell(obs, cellTime)`. Cells carry `data-time` so the
  mobile slider's `_sliderTimes` array picks them up alongside forecast
  cells. Toggle state lives in `localStorage['cs_show_history']`.
  `setShowHistory(val)` syncs the desktop toolbar pill-switch
  (`#desktop-history-switch`) and the Preferences modal checkbox
  (`#pref-show-history`), then anchors the rebuild on the model-overview
  column to keep its viewport-x stable across toggles.
- **Background preload** — `preloadHistoricalData()` fires
  `/api/buoy_historical_context` for every buoy in `CFG.spots` after
  initial render (idle-callback). `_scheduleHistRebuild()` debounces
  re-renders as data arrives.
- **Mobile slider** — `_colIndexFromPct(pct)` and `_pctFromColIndex(idx)`
  switch between two mappings based on `localStorage['cs_show_history']`:
  history-off keeps the legacy "small buoy slot at pct=0" buoy view;
  history-on maps `pct ∈ [0, 1]` linearly across the full timeline so
  pct=0 lands on the oldest historical cell. `_sliderResetToNow()` is
  the canonical "snap to now" action (called by double-tap and on the
  first build via the `_sliderResetDone` one-shot).
- **Touch-action lock** — `.table-scroll *` carries
  `touch-action: pan-y !important` so any descendant cell can't initiate
  a horizontal pan. Combined with `overscroll-behavior: contain`,
  `-webkit-overflow-scrolling: auto`, and `transform: translateZ(0)`
  on `td.spot-cell`, this is the mobile-scroll-stability stack.
- **Buoy modal** — two stacked canvases (`#buoy-popup-canvas-bot` for
  the spectrum on top visually, `#buoy-popup-canvas-top` for the energy
  history below). `_drawBuoyChartTop` renders the energy line;
  `_drawBuoySpectrum` renders the static spectrum at the scrubbed time
  (default x-axis 0–22 s, auto-extends if energy/components reach further).
  `_updateBuoyTimeLabel` and `_updateBuoyInfoStrip` keep the date/time
  label above the charts and the swell readout below in sync on both
  hover (`_attachChartHover`) and scrub (`_buoyScrubApply`).
  Component labels on the spectrum use a four-candidate placement loop
  to avoid overlap.

## Caching architecture (important when debugging staleness)

Data flows through **four** caches; a staleness bug could live in any of
them:

1. **Origin TTL cache** (`cache.py` → `@ttl_cache`) — in-memory dict with
   write-through to `.cache/*.json` on disk. Current per-fetcher TTLs:
   - `fetch_buoy`: 600 s
   - `fetch_buoy_history`: 1800 s
   - GFS waves / wind spot fetchers: 3600 s
   - Wind grids: `@model_aware_cache`, 6 h hard TTL with run-based early
     invalidation on `WIND_UPDATE_HOURS_UTC` (EURO wind 4x/day per
     Open-Meteo's ecmwf_ifs025 cadence)
   - `fetch_cmems_point` (EURO waves): `@model_aware_cache`, 24 h hard TTL,
     invalidated on `MODEL_UPDATE_HOURS_UTC` (~07/19 UTC when CMEMS ANFC
     publishes) — new runs land within one 30-min warmer cycle; upstream
     is hard-capped at 2 cycles/day
   - Tides: month-anchored ~4-month window per station, 45-day TTL
     (`tide._fetch_station_window`), sliced locally per request
2. **CDN edge + browser cache — intentionally DISABLED (2026-07).**
   `_add_cache_headers` sends `no-store` on all `/api/*` GETs and
   `no-cache` on HTML. The old policy table (edge `s-maxage` +
   `stale-while-revalidate` + browser `max-age`) made a new model run
   take 2-3 reloads to appear (SWR served stale while revalidating in
   background; browser max-age then re-served that stale copy). Do not
   re-add edge caching without solving that. Freshness is now bounded by
   the origin TTL cache alone; the warmer keeps origin hits fast.
3. **Frontend in-memory** — `historicalData[buoy_id]`, populated by
   `preloadHistoricalData()` after the initial render; invalidated on
   `refreshAll()` so a manual refresh re-pulls the historical-context
   endpoint in addition to clearing the origin caches.

An installed webapp resumes its frozen page on re-open without reloading,
so `index.html` also has a foreground-refresh hook (`_softRefresh`, on
`visibilitychange`/`pageshow`): when the app becomes visible and the last
successful `loadAll` is >60 s old, it runs `refreshAll({soft: true})` —
the full re-fetch/re-render path minus the rate-limited
`POST /api/refresh` origin bust and the "no new model data" short-circuit.

The background cache warmer (`_cache_warmer_loop` in `app.py`) runs every
1800 s and pre-fetches everything; it piggybacks on the TTL cache, so a
fetcher's TTL must be shorter than 1800 s for the warmer to reliably
refresh it.

To force-clear all caches at runtime: `POST /api/refresh` (rate-limited to
1 call per 30 s per IP).

Fault-tolerance:
- The last-known-good forecast fallback persists to `.cache/lkg_forecast.json`
  so it survives restarts; served with `_status: "stale"` when live fetch fails.
- Partially-populated payloads (`/api/forecast/*`, `/api/wind`) carry
  `_status: "partial"` so the frontend can badge degraded data (badge UI
  pending design approval — see `development-assets/design-demo/index.html`).
- The frontend stashes the last good payload set in
  `sessionStorage['cs_snapshot_v1']` (≤6 h) and instant-paints the table from
  it on reload before fresh fetches land.

## Conventions

- No build step — `index.html` inlines all JS/CSS. Do not introduce a
  framework or bundler.
- No comments unless the *why* is non-obvious. Existing comments are terse
  and justified; match that tone.
- Prefer editing files in place over introducing new modules.
- Python style: follow whatever the file already uses.
- **Dashboard / CSC2 identity**: every CSC2 forecast row must match the
  dashboard byte-for-byte for the same (buoy, valid_utc, model) tuple.
  Anything that feeds training must pass through `waves_cmems` / `waves`
  exactly the way the live dashboard does — no shortcut pulls of raw
  CMEMS or raw GRIB values. The `raw_rows_to_hourly_records` helper in
  `waves_cmems.py` is the canonical entry point for historical EURO
  sources; waves.py `_build_components` is the canonical processor for
  GFS. If CSC2 output disagrees with a dashboard cell for the same
  hour/buoy, something is wrong — not a model difference.

## Launchd jobs on this Mac

Production runs as user-agent plists at `~/Library/LaunchAgents/` —
full setup/troubleshooting detail in `hosting.md`:

- `com.colesurfs.server` — Flask + Waitress on :5151 (log `/tmp/colesurfs.log`)
- `com.colesurfs.autopull` — `git pull origin main` every 90 s
- `com.cloudflare.cloudflared` — tunnel to `surfreport.coleheine.com`
- `com.colesurfs.csc2-logger` — CSC2 forecast logger @ 3 AM + 3 PM ET
- `com.colesurfs.csc2-obs` — CSC2 observation logger every 30 min
- `com.colesurfs.csc2-eval` — daily live-eval pass @ 5 AM ET
- `com.colesurfs.csc2-retrain` — quarterly retrain (see csc2 section)

To reload any service after a code change:
`launchctl kickstart -k gui/$(id -u)/<label>`.
