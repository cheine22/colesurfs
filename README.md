# colesurfs · v1.8.1

© 2026 Cole Heine. All rights reserved. — [LICENSE](./LICENSE)

A surf forecast dashboard for the NY / NJ / New England coast. Designed to provide the best tools for the experienced surfer & surf forecaster to anticipate windows of good waves and plan surf sessions. Guiding principle: interpretation of the agreement between the different models provides the most accurate predictive information. Pulls live buoy data from NOAA and swell/wind model forecasts from Open-Meteo and Copernicus Marine (CMEMS), then presents everything in one scrollable view: a color-coded swell table synced to an animated wind map, with per-spot tide predictions and wind condition ratings.

Flask backend, vanilla HTML/CSS/JS frontend. The CMEMS EURO path (C-EURO) authenticates via the `copernicusmarine` CLI; everything else uses free unauthenticated NOAA / Open-Meteo endpoints.

---

## Features

- **Easy toggling between EURO & GFS wave forecasts** — hourly or 3-hour swell table with up to 3 swell partitions per cell, color-coded by swell category
- **Model concordance** — "Model Agreement" badge appears when EURO and GFS predict the same swell category
- **Customized swell rating scale** — 7 hierarchical tiers (Flat / Weak / Fun / Solid / Firing / Hectic / Monstro) per swell (current or modeled) based on swell size and period
- **At-a-glance swell forecast evaluation** (new in v1.7) — per-spot count of days in the 10-day forecast where GFS and EURO both classify the primary swell as fun-or-better for ≥6 daytime hours. Cell colour tracks the best `min(GFS, EURO)` window across the forecast.
- **Historical-data mode** (new in v1.7) — toggle in the toolbar (desktop) or Preferences modal (mobile) reveals a -240 h buoy-observation strip to the left of the Fun+ Days column, with a ✓ glyph on cells where both models' archived forecasts agreed with the observed classification. Cadence matches the resolution toggle; data preloads in the background from CSC2 archives.
- **NOAA buoy readings** — live wave height, dominant period, and direction per buoy
- **Historical buoy popup** — BUOY HISTORY button opens a 3-day modal with two stacked charts: a live frequency spectrum at the scrubbed time (top) and energy-over-time (bottom). Date/time label above the charts, swell readout below, dotted-line hover indicator on desktop, touch scrubber on mobile.
- **Tide predictions** — NOAA CO-OPS harmonic predictions per spot, with Surfline-matched time corrections
- **Individual swell components** — spectral analysis producing primary + secondary swell partitions from raw NDBC spectral files
- **Animated wind map** — Leaflet.js with a custom HiDPI canvas particle system (Windy-style), synced to the swell table by hover time. Retina-aware tiles and rendering
- **Customized wind condition rating scale** — 6 hierarchical tiers (Glassy / Groomed / Clean / Textured / Messy / Blown Out) per spot based on wind direction relative to the measured coast angle and speed
- **Smart API caching** — model-run-aware cache that skips API calls when cached data is still from the latest model run
- **Smart refresh** — refresh button checks for new model data before clearing caches; shows toast if no new data available
- **YAML-driven region config** — all regions, buoys, and spots defined in `regions.yaml`; adding a new region requires no code changes
- **Mobile-optimized layout** — responsive portrait layout with velocity-based time scrubbing (iOS-style precision control). Double-tap the slider snaps the Fun+ Days column flush against the sticky spot column.

---

## Data Sources

All data is fetched from free or free-tier public services:

- **NOAA NDBC** — live buoy observations and spectral swell data (updated every 30 min); yearly stdmet archives back to 2021 used for NDBC backfill
- **Copernicus Marine (CMEMS)** — ECMWF WAM ANFC with swell partitions (VHM0_SW1/SW2, VTM01_SW1/SW2, VMDR_SW1/SW2), the EURO source for both the dashboard and CSC2. Live fetch via `copernicusmarine` (free `copernicusmarine login` credential at `~/.copernicusmarine/.copernicusmarine-credentials`)
- **Google Earth Engine — `COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H`** — cycle-preserving archive of the same CMEMS product, used by CSC2's historical backfill because CMEMS itself overwrites past cycles. The GEE mirror ingests one cycle per day starting 2025-04-28; see `csc2/gee_backfill.py`. Free for noncommercial use (Community Tier: 150 EECU-hours/month)
- **AWS Open Data — `s3://noaa-gfs-bdp-pds/`** — NOAA GFS-Wave GRIB2 archive with swell partitions back to 2021-04 used by CSC2's historical backfill; byte-range fetches via `.idx` sidecars
- **Open-Meteo Marine API** — live GFS-Wave partition forecasts for the dashboard and live logger (GFS stream only; EURO migrated off Open-Meteo in v1.5)
- **Open-Meteo Forecast API** — ECMWF IFS and GFS wind model forecasts
- **NOAA CO-OPS** — harmonic tide predictions per spot with Surfline-calibrated time corrections

---

## In beta: CSC2 — Colesurfs Correction v2 (lead-time-aware forecast correction)

**Goal.** A forecast-correction model that predicts the primary + secondary
swells (height, period, direction) at a fixed set of NDBC buoys using patterns
in the difference between the EURO and GFS model data across lead time, plus
the time of year. Outputs are produced in the same dashboard-ready units as
the `EURO` and `GFS` buttons, so a future `CSC2` button drops in alongside them.

**Scope.** Eight buoys total, split into two tracks that share code and
architecture but nothing else:

- **East (user-facing)** — 44013 Boston, 44065 NY Harbor Entrance, 44097 Block Island Sound, 44091 Barnegat (NJ), 44098 Jeffrey's Ledge (NH)
- **West (silent, parallel)** — 46025 Santa Monica Basin, 46221 Santa Monica Bay, 46268 Topanga Nearshore

**Identity with the dashboard.** Every forecast value that feeds training and
live correction passes through the exact same processing pipeline as the main
dashboard (`waves_cmems.py` and `waves.py`: 5.0 s Tm01 filter, `Tm01 × 1.20`
Tp-rescale for CMEMS partitions, energy-sorted top-2 partitions, no
combined-sea fallback). If a CSC2 cell disagrees with the dashboard's EURO or
GFS cell for the same hour and buoy, something is wrong — not a model
difference.

**Data sources** (all kept strictly consistent with what the dashboard serves):

| stream            | live (going forward)                                | historical backfill                                                              |
|-------------------|-----------------------------------------------------|----------------------------------------------------------------------------------|
| EURO (CMEMS)      | `com.colesurfs.csc2-log` @ 3 AM + 3 PM ET           | Google Earth Engine `COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H` (2025-04 → today) |
| GFS (Open-Meteo)  | same plist, same schedule                           | AWS `s3://noaa-gfs-bdp-pds/` byte-range GRIB2 fetch (scope A: 2025-04 → today)   |
| Buoy observations | `com.colesurfs.csc2-obs` @ every 30 min             | NDBC stdmet yearly archives (2021 → today)                                       |

**Model architectures.** Two tiers reported side-by-side on the eval page:

- *CSC2+baseline* — per-[buoy × lead-hour × variable] linear bias correction. Simple floor; no learned interactions.
- *CSC2+ML* — gradient-boosted trees (LightGBM) over features: raw EURO partitions, raw GFS partitions, EURO–GFS deltas per lead, sin/cos day-of-year, buoy identity, and **lead hour as a first-class feature** (since the archive preserves full 0..+240 h lead structure for every cycle). One model per output variable.

**Naming.** "CSC2" alone refers to the *training dataset* (paired GFS + EURO + buoy obs). A trained model instance is always named:

```
CSC2+{baseline|ML}_{YYMMDD}_{coverage}_v{N}
```

`YYMMDD` = train date (UTC, sorts lexicographically); `coverage` = fraction of 365 with ≥1 paired GFS + EURO + **spectral-swell-buoy** day pooled across the 5 east buoys, rounded to 0.01 (>1.0 once we cross into year 2). "Spectral-swell-buoy" means a hour where at least one of partition=1 (primary swell) or partition=2 (secondary swell) — produced by the same `_spectral_components` decomposition the dashboard uses — has Hs/Tp/Dp; partition=0 (combined sea, basic NDBC stdmet WVHT/DPD/MWD) does NOT count, because the model targets sw1/sw2 to match dashboard cells. `v{N}` = architecture/hyperparameter variant for same-day comparisons. Example: `CSC2+ML_260424_0.77_v2`. Weights land in `.csc2_models/east/<full-name>/`. The west track uses identical naming under `.csc2_models/west/`.

**Evaluation.** [`/csc`](http://localhost:5151/csc) renders two tables with a
buoy picker (individual or combined):

- *Table 1 — Traditional metrics*: MAE / RMSE / bias for primary and secondary swell Hs, Tp, direction. One row per (variable × statistic), four columns (Raw EURO, Raw GFS, CSC2 baseline, CSC2 model). Color-coded best → worst per row.
- *Table 2 — Surfer metrics*: Sensitivity / specificity / PPV / NPV for the dashboard's categorizer, stratified per category (FUN, SOLID, FIRING, HECTIC, MONSTRO) plus a combined FUN-or-better class. Same four-column layout.

An "Archive accumulation" panel at the top of the page stays visible even
after training, showing per-buoy EURO cycles, GFS cycles, and **paired cycles**
(init times where both model forecasts *and* matching buoy observations
exist — the minimum condition for a trainable sample). The page also scaffolds
a live forecast row showing CSC2 vs EURO vs GFS for the selected buoy out to
+240 h, activated once the model is trained.

**Cadence to first training.** First model trains once we've accumulated
paired cycle coverage for >=90% of days of the year; target is 60 months so the model can learn
seasonal pattern changes. The GEE backfill brings us to ~12 months of
lead-resolved CMEMS coverage at kickoff; GFS and NDBC backfills cover
the same window with full lead structure; live loggers top both up daily.

**West-coast track** uses identical architecture; models train silently,
artifacts land in `.csc2_models/west/`, and nothing is surfaced on the main
dashboard until explicitly promoted.

---

## Swell Categorization

The app uses a **7-category 2D lookup system** based on both wave height (ft) and period (s), capturing how a given wave height reads very differently at 6 s vs 16 s.

| Category    | Description                                                                        |
| ----------- | ---------------------------------------------------------------------------------- |
| **FLAT**    | Nothing to surf                                                                    |
| **WEAK**    | Marginal conditions but something to surf                                          |
| **FUN**     | Promising swell for a fun session                                                  |
| **SOLID**   | Likely overhead surf, high quality waves on tap at the right place                 |
| **FIRING**  | Longer period surf with good size. Potential for absolutely firing conditions.     |
| **HECTIC**  | Potential for maxed out spots, difficult to find a decent wave with so much energy |
| **MONSTRO** | Good luck finding a break that can handle it                                       |

Thresholds are defined per period band in `swell-categorization-scheme.toml`.

---

## Wind Condition Rating

In regional mode, each spot is classified by surf-quality based on wind direction relative to the coast and sustained wind speed. The 6 tiers are evaluated hierarchically (first match wins):

| Rating | Color | Condition |
|---|---|---|
| **Glassy** | Green | Offshore, sustained < 9 mph |
| **Groomed** | Green | Offshore, sustained > 20 mph |
| **Clean** | Green | Offshore at any speed in between, or any direction with sustained < 3 mph |
| **Textured** | Gold | Sideshore + sustained < 15 mph, or onshore + sustained < 8 mph |
| **Messy** | Blue | Sideshore + sustained < 18 mph, or onshore + sustained < 13 mph |
| **Blown Out** | Grey | Everything else |

Wind direction zones are defined relative to each spot's measured shore normal: offshore (≤ 32° from offshore direction), sideshore (32°–115°), onshore (> 115°). Thresholds are tunable per-spot via `/tuner`; live values are stored in `wind-categorization-scheme.toml`.

---

## Local data layout (not committed to Git)

All CSC2 training inputs and model artifacts live outside Git. `.gitignore`
covers every directory below:

```
colesurfs/
├── .csc_data/                 # legacy observation archive, preserved (buoy-only, model-agnostic)
│   ├── observations/          # NDBC stdmet yearly archives (2021 → today), written by csc2.ndbc_backfill
│   └── live_log/observations/ # 30-min live obs, written by csc2.obs_logger
├── .csc2_data/                # CSC2 forecast archive — fresh, lead-resolved, per-cycle parquet
│   ├── forecasts/
│   │   ├── model=EURO/buoy=<id>/year=Y/month=M/cycle=YYYYMMDDTHHZ.parquet
│   │   └── model=GFS/buoy=<id>/year=Y/month=M/cycle=YYYYMMDDTHHZ.parquet
│   ├── logs/                  # per-job append-only text logs
│   └── archive_status_cache.json   # cached payload for /api/csc2/archive_status
└── .csc2_models/              # trained model weights (materializes when first model fits)
    └── east/<version>/ …
    └── west/<version>/ …
```

Why not Git?

- **Size** — ~50 MB today, grows by ~300 MB/year once the live loggers and
  GFS backfill both settle. Not a good fit for Git history.
- **Reproducibility** — every row is deterministically rebuildable from public
  sources (CMEMS via GEE, NOAA GFS-Wave via AWS Open Data, NDBC stdmet
  archives). The backfills are idempotent: `python -m csc2.gee_backfill`,
  `python -m csc2.aws_gfs_backfill --start YYYY-MM-DD`, `python -m csc2.ndbc_backfill`.
  A fresh clone on a new machine just re-runs them.
- **Mutability** — shards get rewritten when backfills re-run or when the
  live loggers fire. Git isn't the right store for append-only time series.

---

## Known Limitations

- No wave breaking / beach angle correction — offshore buoy and model data only. This model is not meant to be a one-stop-shop to compare the surf height at different locations, it is meant to provide a data that can be verified on the day through the buoy for session planning based on local knowledge. 
- The wind particle field is drawn at the model grid resolution (144 points at 4° spacing), not interpolated to a finer mesh
- Tide corrections were calibrated on a single date (2026-04-01) against Surfline's spot data and are fixed constants

---

## Changelog

### v1.8.1
- **Outage-modal stops firing on transient cache gaps.** New last-known-good fallback in `app.py:api_forecast` — if a fresh fetch returns empty for either EURO or GFS (e.g. mid-cache-rebuild after `POST /api/refresh`, or a brief upstream blip), the endpoint serves the previous successful response from a process-memory `_last_known_forecast` dict instead of `{}`. The dashboard's outage modal only fires when we've **never** had a populated response since the server started — i.e. genuine sustained upstream failure.
- **Model-run indicator now reflects EURO + GFS independently.** GFS publishes 4 cycles/day (00/06/12/18 Z); EURO publishes 2 cycles/day (00/12 Z); they very often disagree on which run is current. `_refreshStatus` now fetches both `/api/status?model=EURO` and `/api/status?model=GFS` in parallel. When the runs are aligned the indicator collapses to a single readout (`Apr 25 12Z · next run in ~3h`); when they diverge it expands to show both (`EURO Apr 25 00Z · GFS Apr 25 12Z`) so the divergence is impossible to miss. Smart-refresh still keys off the active model's run.
- **Buoy modal — `BUOY HISTORY` button renamed to `BUOY SPECTRA`** (desktop toolbar). Modal contents unchanged on both desktop and mobile.
- **Buoy modal — "Max single swell" callout** in the spectrum chart's top-right shows the largest-energy single swell partition observed across the whole 3-day window. Pre-computed once on data load via new `_findMaxEnergyComponent`; doesn't change as the user scrubs.
- **Buoy modal — spectrum chart Y axis switched to wave-energy index `H × T²` (ft·s²)** from the previous wave-energy-flux `H² × T` (kW/m). Same X axis, same partition labels, but longer-period swells now dominate the visual hierarchy in line with the surfer-energy intuition. Unified across `buoy.py` (`_parse`, `_spectral_components`, `_parse_spec`, `fetch_buoy_history`) and the JS callout / chart so every place wave energy is computed uses the same formula.
- **Buoy modal — spectrum chart 2× taller** (220 → 440 px desktop, 140 → 280 px mobile); modal `max-height: 96vh` + internal `overflow-y: auto` as a safety net so the page itself doesn't scroll on phones.
- **Buoy modal — spectrum Y-axis stays constant across the 3-day window.** New `_computeSpectrumMaxFlux` pre-scans every record's spectrum on data load and locks the Y-axis ceiling so peaks at quiet times correctly read smaller than peaks at busy times.
- **BUOY NOW cell tap opens the modal pre-loaded to that buoy.** The previous mobile blocker — `pointer-events: none` blanket rule on data cells — is now bypassed via `td.cell.buoy-now[onclick]` selector that re-enables only the cells with a click handler.
- **Audit fix — snap-to-hour mismatch** between `archive_status` and `train.py` (rounded `:30:00` differently). Both now round half-up consistently.
- **Mobile header swap** — model-run-tag moves into the header right slot (where PREFERENCES used to live), PREFERENCES moves to the bottom bar; stacked layout mirrors the original `mob-meta-run` / `mob-meta-next` styling.
- **Main-page `Fun+ Days` cells** now show `{count}/{forecast-window-days}` and gate on a wind filter (≥1 spot in the same `buoy_region` has Glassy/Groomed/Clean/Textured wind at the same 3-hour window).
- **Desktop `CSC2 (BETA)` button** in the toolbar between History toggle and the status bar — direct link to `/csc`.

### v1.8
- **CSC2 leaves "data collection only" — models are trained, ranked, and live on `/csc`.** End-to-end pipeline shipped: `csc2/train.py` (paired-data loader + CSC2+baseline (per-(buoy × lead × variable) additive bias) + CSC2+ML (LightGBM, 8 boosters) + holdout eval); `csc2/registry.py` (composite-skill ranking, weighted **sw1 height 0.25 / sw1 period 0.25 / sw1 dir 0.05 / sw2 each 0.05 / surfer FUN-OR-BETTER F1 0.30**, with a min-test-rows gate so noisy small-holdout scores don't take TOP); `csc2/predict.py` (live inference matching the dashboard fallback path byte-for-byte). The `/csc` page now surfaces the **#1 performer + 2 most recent additional models** with a 10-day live forecast chart, surfer metrics (Sens/Spec/PPV/NPV per FUN/SOLID/FIRING/HECTIC/MONSTRO + combined FUN-OR-BETTER), traditional MAE/RMSE/bias, and the running archive accumulation panel.
- **Spectral backfill closes the partition=1/2 obs gap on every buoy.** Three new modules:
  - `csc2/ndbc_spectral_backfill.py` — yearly + monthly + realtime NDBC swden+swdir → run through the dashboard's `_spectral_components` for byte-identical decomposition.
  - `csc2/cdip_spectral_backfill.py` — CDIP THREDDS `historic.nc` + `_rt.nc` for buoys NDBC doesn't archive (44091 USACE, 44097 UConn, 44098 UNH-NERACOOS, 46221 SM Bay).
  - Combined east-pool paired-cycle coverage: 36 → ~265-286 per buoy. East-pool calendar coverage **0.77 → 0.80** (`coverage` figure in model names now reflects partition=1/2-gated days, not partition=0).
- **Quarterly retrain + daily live-eval scheduled.** Two new launchd jobs: `com.colesurfs.csc2-retrain` (1st of Mar/Jun/Sep/Dec at 04:00 — wipes archive cache, recomputes coverage, writes both architectures as `_v1` of the date), `com.colesurfs.csc2-eval` (daily 05:00, runs every saved model on cycles that post-date its training, appends rolling-30d skill to `.csc2_data/live_eval/<model>.parquet`).
- **CSC2 model naming convention.** `CSC2+{baseline|ML}_{YYMMDD}_{coverage}_v{N}`. `coverage` = east-pool fraction of 365 with ≥1 paired GFS+EURO+spectral-buoy day. `vN` = same-day architecture variant. Documented in CLAUDE.md and the new `/csc-model` flowchart page.
- **`/csc-model` documentation page.** Single-page docs explaining anatomy of the data, processing pathway, model architectures, evaluation metrics, and naming convention with a horizontal SVG flowchart. Theme-aware (light/dark, mirrors main dashboard's `wave-theme` localStorage). Linked from the `/csc` subtitle.
- **Buoy modal upgrades.**
  - **Tap on BUOY NOW cell opens the modal preloaded to that buoy.** Previously only the BUOY HISTORY toolbar button could open the modal (defaulting to the regional buoy). The blocker was a leftover mobile-CSS `pointer-events: none` rule on `.swell-table td.cell` that made data cells inert; the new selector `.swell-table td.cell.buoy-now[onclick]` re-enables only the cells that carry an open-modal handler.
  - **Spectrum chart Y axis = wave energy flux (kW/m).** Replaced the spectral-density / per-bin-Hs view with `(ρg²/4π)·E·Δf·T` per bin, which captures the surfer-energy intuition that long-period swells punch above their Hs weight. Y-axis ticks read directly in kW/m.
  - **Constant Y-axis ceiling across the 3-day window.** New `_computeSpectrumMaxFlux(records)` pre-scans every record's spectrum on data load and locks the Y-axis to the global max so peaks at quiet times correctly read smaller than peaks at active times.
  - **Spectrum chart 2× taller** (220 → 440 px desktop, 140 → 280 px mobile). Modal `max-height: 96vh` + `overflow-y: auto` as a safety net so the page itself doesn't scroll on phones.
- **Main dashboard tweaks.**
  - **Fun+ Days format** changed from a single `count` to **`{count}/{forecast-window-days}`** (e.g. `5/10`).
  - **Fun+ wind filter.** A 3-hour window only counts as Fun+ if at least one spot in the same `buoy_region` has Glassy/Groomed/Clean/Textured wind at the same time per `windCondition()`. Honest-empty fallback: if regional wind data isn't loaded yet the filter no-ops so the column doesn't go blank during partial load.
  - **Desktop CSC2 (BETA) button** in the toolbar between History toggle and the status bar — direct link to `/csc`, mobile bottom-bar already had a CSC entry inside Preferences.
  - **Mobile header now shows the model run-time + next-run ETA** (was previously in the bottom bar). The PREFERENCES button moves to the bottom bar in the slot the run-tag vacated. Stacked layout matches the original `mob-meta-run` / `mob-meta-next` styling exactly.
- **Buoy + forecast logger expansion to all 8 CSC2 buoys.** West-coast buoys (46025, 46221) get the same NDBC + CDIP backfill treatment as east; daily 30-min obs collection already covered them. Post-backfill west-pool paired counts: **231 (46025) / 237 (46221)**.
- **Audit fix: time-snap mismatch.** `archive_status._snap_to_hour` and `train.py:_snap_to_hour_iso` previously rounded `:30:00` differently (down vs up), causing silent off-by-hour mis-joins. Both now round half-up consistently.
- **Mobile CSC2 page polish.** Tables wrap into `display: block; overflow-x: auto` containers so wide comparison tables (active models, Table 1, Table 2 categories, archive) scroll horizontally inside the viewport instead of overflowing the page. Sticky 44px header (mirroring main dashboard's `.app-header`), sticky footer with buoy-picker + made-by-cole-and-claude credit, light/dark theme toggle, archive section moved to the bottom.

### v1.7.1
- **Fix: GFS empty cells beyond day 5** — `_parse_response` in `waves.py` now falls back to the combined sea state (`wave_height` / `wave_peak_period`) when GFS swell partitions are absent (GFS drops them beyond ~5 days). Previously those timesteps returned null for height/period; they now show the combined Hs converted to feet with Tp. Direction is included when available; `components` remains empty to preserve downstream logic that distinguishes partition-backed cells from combined-sea cells.

### v1.7
- **Fun+ Days column** — new column between the spot name and BUOY NOW that counts forecast days where GFS and EURO both classify the primary swell as FUN-or-better for at least 6 daytime hours (two 3-hour windows). Cell background reflects the best `min(GFS_cat, EURO_cat)` across all daytime 3-hour windows in the forecast horizon. Analysis always runs at a 3-hour stride so the count is stable across the Hourly/3-Hour toggle. Region mode shows the column only on the dedicated buoy row; spot rows render blank cells.
- **Historical buoy strip** — new toggle-gated `-240 h` section to the left of the Fun+ Days column, with cells drawn from the existing NDBC realtime2 file (no extra API calls). Each cell finds the closest-to-hour observation (tolerance ±30 min at 1 h cadence, ±90 min at 3 h). A ✓ glyph in the top-right marks cells where both archived EURO and GFS forecasts at that hour matched the observed classification. For CSC2-scoped buoys (44013, 44065, 44097, 44091, 44098) the glyph is sourced from local parquet archives; non-CSC2 buoys render obs-only.
- **Toolbar reorganisation (desktop)** — `HISTORY OFF | ON` pill-switch sits next to the Resolution toggle; `HOURLY | 3-HOUR` restyled to match the EURO/GFS pill-switch (generic `.sw-btn.active` rule in place). Preferences-modal checkbox is now mobile-only; state syncs in both directions across toolbar, modal, and `localStorage['cs_show_history']`.
- **Scroll-stable history toggle (desktop)** — `setShowHistory` anchors the rebuild on the Fun+ Days column so its screen-x doesn't jump when 80 historical cells are added or removed to its left. Measured viewport-x held constant across on/off cycles.
- **Buoy modal rebuild** — two stacked charts (frequency spectrum on top, energy-over-time below), date/time label above the charts, swell readout below, dotted-line hover indicator on both platforms. Range shortened from 5 days to 3. Desktop tooltip removed; info now lives in the persistent time-label and info-strip elements on both platforms. Spectrum line uses `lineWidth = 1.0`; dot radius 1.25.
- **Frequency-spectrum chart** — period ascending left→right (long period on the right, surfer convention). Default x-axis range 0 s–22 s; auto-extends when spectral bins or decomposed components carry energy beyond 22 s. Component markers draw with a contrast-pill background (`var(--bg1)` + `var(--border0)`) so labels read clearly over the translucent fill. Labels now try four candidate placements (right-above, left-above, right-below, left-below) and pick the first that fits in bounds without colliding with an already-placed label rect — no more overlap at close period spacing (verified at 1.6 s gap).
- **Mobile slider behaviour** — with history on, `_sliderTimes` now includes historical timestamps so the slider maps linearly across `[oldest history … latest forecast]`. Dragging all the way left shows the -240 h column (oldest observation). Double-tap calls `_sliderResetToNow()`, which scrolls the Fun+ Days column flush against the sticky spot column and positions the handle at the history/forecast boundary (~53 % on a history-on slider). Fresh builds triggered by async historical-data arrival (`_scheduleHistRebuild`) no longer yank the table back to "now" mid-scroll — the one-shot `_sliderResetDone` flag preserves user scroll state after the first reset.
- **Mobile horizontal-scroll lock** — `touch-action: pan-y !important` now cascades to every descendant of `.table-scroll` (not just the container), with `overscroll-behavior: contain`, `-webkit-overflow-scrolling: auto`, and `transform: translateZ(0)` on the sticky spot column. Vertical swipes no longer generate horizontal drift on iOS, and touches starting on a data cell can't initiate a horizontal pan gesture.
- **Fun+ Days typography** — cell count rendered at 18 px, `font-weight: 800`, centred horizontally and vertically. Header reads "Fun+<br>Days".
- **Modal content swap** — logo modal holds welcome text, the renamed "swell categories" legend, wind classifications, tips, and a new **Data provenance** section with the dl attributions (moved from its previous spot). The old About Me modal is renamed to **Preferences** with title `∿colesurfs · preferences`; button order is REFRESH → CSC → LIGHT → TUNER → show-historical-data toggle. Logo modal is scrollable (`max-height: 75vh; overflow-y: auto`). A new line reads "for my main site, visit coleheine.com" with the link inline-styled to match the GitHub link.
- **Historical-context backend** — new `GET /api/buoy_historical_context?station_id=<id>&days=10` endpoint returns records with `observed_cat`, `model_agreement` (`true`/`false`/`null`), and the existing spectral-components array. `fetch_buoy_history` default bumped 5 → 10 days and now attaches a raw `spectrum` field (`[[freq_hz, energy_density_m2/Hz, direction_deg | null], …]`) per record, sourced from the same `_parse_spectral_file_all_rows` output we already parse for component decomposition. **No additional NDBC API calls** — the extra range and spectrum data come from bytes already on disk.

### v1.6
- **CSC2 housekeeping sweep** — following a three-agent codebase audit (redundancy, efficiency, documentation), applied every green-light finding in nine coordinated batches while the GFS backfill kept running
- **Unified obs schema** — live-log observations now write `hs_m` (SI, matches stdmet historical) instead of `hs_ft`; existing 34 live-log shards migrated in place. `archive_status` dropped its per-source Hs fallback
- **Fixed latent null-partition-0 bug** — `obs_logger._rows_for_buoy` was looking up `height_ft`/`wvht_ft` keys that `fetch_buoy` doesn't return (real key is `wave_height_ft`), so partition-0 rows had been silently all-null. Mean paired-cycle coverage lifted 163.9 → 165.5 after the fix
- **Canonical ISO timestamps** — all three writers (`logger.py`, `obs_logger.py`, `ndbc_backfill.py`) now emit `Z`-suffix UTC strings; legacy `+00:00` rows converge on rewrite via a re-normalization step in dedup
- **Public shard API** — `csc2/logger.py` helpers renamed from `_shard_path` / `_records_to_rows` / `_write_rows` to public names, with underscored aliases retained so the in-flight GFS backfill subprocess keeps working
- **Obs fast-path** — `_append_dedup` now skips parquet rewrites entirely when the incoming `(valid_utc, partition)` set is already on disk (NDBC updates at 10-min cadence while the logger polls every 30 min, so most ticks are duplicates)
- **`ndbc_backfill` parallelism** — yearly archive pulls fan out across a thread pool (default 8 workers) instead of serializing
- **Archive-status sentinel fast-path** — cache staleness check resolves in O(1) via `.csc2_data/forecasts/.last_write` sentinel, with the original full-tree rglob kept as a safety net
- **Plist annotation** — both csc2 plists now carry a comment explaining the asymmetric `RunAtLoad` choice (false for cycle logger to avoid cycle-shard collision; true for obs logger since rows dedup harmlessly)

### v1.5.1
- **CSC2 scaffold** — fresh forecast-correction stack replaces legacy `csc/`. Fixed buoy scope (5 east + 3 west), identity-with-dashboard contract, cycle-preserving historical backfills (CMEMS via Google Earth Engine, GFS via AWS `noaa-gfs-bdp-pds`, NDBC stdmet yearly archives), live loggers (`com.colesurfs.csc2-log` @ 3 AM + 3 PM ET for forecasts; `com.colesurfs.csc2-obs` every 30 min for buoys), and `/csc` eval page with archive-accumulation table, model definitions, and per-category surfer metrics
- **EURO dashboard tightening** — keeps v1.5's CMEMS migration, honest-empty policy, and `Tm01 × 1.20` Tp-rescale; the single `EURO` button now stays visible in place with no `C-EURO`/`OM-EURO` vestiges
- **Local-only data layout** — new `.csc2_data/` forecast archive is gitignored alongside the legacy `.csc_data/`; docs list what lives where and why none of it is in Git

### v1.5
- **EURO wave forecast migrated to Copernicus Marine (CMEMS)** — Open-Meteo's `ecmwf_wam025` endpoint returns `null` for every swell-partition variable, so the dashboard's OM-EURO column had been silently serving combined-sea values as "primary swell" (wind chop included). CMEMS ANFC publishes real `VHM0_SW1` / `VHM0_SW2` spectral partitions. `/api/forecast/EURO` now calls CMEMS; Open-Meteo ECMWF-WAM was removed from the site
- **Cache warmer** — 30-minute warmer pulls CMEMS for every region buoy in parallel (~11 s wall clock for 7 buoys). 24-hour TTL + `skip_none=True` keeps the last-known-good payload during outages
- **Period convention aligned to Tp** — CMEMS publishes partition period as `VTM01` (spectral mean m1); we scale by 1/0.83 ≈ 1.20 before display so numbers line up with Windy, Surfline, and buoy DPD
- **Partition filter** — the "below 6s is noise" cutoff dropped to 5.0 s (equivalent to Tp ≥ 6 s) and now applies consistently to every wave source
- **No more combined-sea fallback** — if no swell partition passes the filter, the cell stays empty. Previously the table would silently fall back to combined Hs/Tp/direction (including wind chop), which matched neither Windy's partition display nor what a buoy DPD reports
- **Provenance moved** — data-source attribution now lives in the logo-tap info modal; outage modal no longer links to external issue trackers
- **Dependencies** — `copernicusmarine>=2.4`, `xarray`, `netCDF4`

### v1.4
- **CSC v1 (Colesurfs Correction)** — primary-swell bias-correction model; superseded by CSC2 in v1.5.1

### v1.3
- **Batched wave forecast API** — all spot forecasts fetched in a single multi-location Open-Meteo call (7 calls → 1); cold-cache load time dropped from ~14 s to ~2–3 s
- **Server-side sunrise/sunset** — computed via `astral` on the backend; eliminates the browser-side Open-Meteo dependency for day/night column shading
- **Parallel backend fetches** — `ThreadPoolExecutor` for buoy and spot wind fetches
- **Response compression** — `flask-compress` adds gzip/brotli on all responses; Cache-Control + stale-while-revalidate headers for Cloudflare edge caching
- **Background cache warming** — server pre-fetches all data every 30 min so user requests always hit warm cache
- **Disk-based cache persistence** — cache survives server restarts
- **Inlined `/api/config`** — config embedded in the HTML template at render time, saving one round-trip on load
- **Batch fallback** — if the batched wave API call fails, falls back to per-spot fetches

### v1.2.1–v1.2.5
- **Outage detection & modal** — fixed amber top banner evolved into a centered modal that dynamically reports PARTIAL or FULL outages with affected data types, auto-clears on successful reload, and skips the smart-refresh "no new data" check while visible. Wind API gained a `_RATE_LIMITED` sentinel and 30-min negative-cache TTL; failed wind lookups retry via `skip_none=True`. (External-issue-tracker link removed in v1.5.)
- **Photography link & info-modal polish** — COLE HEINE PHOTOGRAPHY button added to the logo info modal; refresh button text standardized to all caps.

### v1.2
- **Smart refresh** — refresh button checks `/api/status` for new model data before clearing caches; shows toast notification if no new run is available
- **Historical buoy popup** — BUOY HISTORY button (desktop controls bar + mobile bottom bar) opens a 5-day wave energy graph (height × period²) with full-opacity swell-category background coloring, category border lines, spectral component tooltips on hover, and an in-popup buoy selector dropdown; in regional mode jumps directly to the active region's buoy; changing buoy in selector also enters regional mode for that buoy
- **Mobile buoy scrubber** — draggable slider below the buoy chart on mobile for scrubbing through the 5-day history with timestamp label, tooltip, and vertical indicator line on the chart
- **About Me modal** — new modal accessible from mobile header button and desktop footer; includes welcome text, GitHub link, swell/wind color legend, and usage tips
- **Mobile header** — replaced LIVE clock with larger ABOUT ME button (75% taller header)
- **Theme toggle moved** — light/dark toggle moved from mobile bottom bar to the info modal (logo tap); desktop footer reordered to: side-by-side, dark/light, about me
- **Wind particle rendering** — replaced `destination-in` alpha compositing with explicit per-particle trail history; eliminates ghost trail artifacts on dark backgrounds caused by 8-bit alpha quantization floor
- **Wind particle coverage** — particles in regional mode now spawn across the full visible map area (was limited to a small bounding box around observation points); IDW interpolation cutoff increased to 4° for full-map coverage
- **Backend: buoy history API** — new `GET /api/buoy_history/<station_id>` endpoint returns 5 days of historical buoy readings with spectral swell components, cached for 30 minutes
