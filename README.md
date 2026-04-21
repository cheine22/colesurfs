# colesurfs · v1.5.1

© 2026 Cole Heine. All rights reserved. — [LICENSE](./LICENSE)

A surf forecast dashboard for the NJ / NY / New England coast. Pulls live buoy data from NOAA and wave/wind model forecasts from Open-Meteo and Copernicus Marine (CMEMS), then presents everything in one scrollable view: a color-coded swell table synced to an animated wind map, with per-spot tide predictions and wind condition ratings.

Flask backend, vanilla HTML/CSS/JS frontend. The CMEMS EURO path (C-EURO) authenticates via the `copernicusmarine` CLI; everything else uses free unauthenticated NOAA / Open-Meteo endpoints.

---

## Features

- **NOAA buoy readings** — live wave height, dominant period, and direction per buoy
- **Individual swell components** — spectral analysis producing primary + secondary swell partitions from raw NDBC spectral files
- **EURO & GFS wave forecasts** — hourly or 3-hour swell table with up to 3 swell partitions per cell, color-coded by swell category
- **Animated wind map** — Leaflet.js with a custom HiDPI canvas particle system (Windy-style), synced to the swell table by hover time. Retina-aware tiles and rendering
- **Model switch** — toggle between ECMWF (EURO) and GFS wave + wind models
- **Regional mode** — click a buoy row to zoom into its surf spots: per-spot hourly wind forecasts and tide data
- **Tide predictions** — NOAA CO-OPS harmonic predictions per spot, with Surfline-matched time corrections
- **Wind condition rating** — 6 hierarchical tiers (Glassy / Groomed / Clean / Textured / Messy / Blown Out) per spot based on wind direction relative to the measured coast angle, sustained speed, and effective gust
- **Model concordance** — "Model Agreement" badge appears when EURO and GFS predict the same swell category
- **Smart API caching** — model-run-aware cache that skips API calls when cached data is still from the latest model run
- **Light/dark mode** — system-aware with manual toggle
- **Side-by-side mode** — table + map split; always on for desktop, portrait layout for mobile
- **Mobile-optimized layout** — responsive portrait layout with velocity-based time scrubbing (iOS-style precision control)
- **Mobile-specific map centers** — per-region map framing tuned for portrait aspect ratio
- **Historical buoy popup** — BUOY HISTORY button opens a 5-day wave energy graph with spectral component tooltips, swell-category background coloring, and a buoy selector dropdown; in regional mode jumps directly to the active region's buoy
- **Mobile buoy scrubber** — draggable slider below the buoy chart on mobile for scrubbing through the 5-day history with tooltip and vertical indicator line
- **About Me modal** — accessible from mobile header and desktop footer; includes swell/wind color legend and usage tips
- **Smart refresh** — refresh button checks for new model data before clearing caches; shows toast if no new data available
- **Mobile info popup** — tapping the logo on mobile opens a modal with a manual refresh button, current API/swell data summary, and version number
- **Version display** — version number shown in desktop footer, mobile info popup, and About modal
- **YAML-driven region config** — all regions, buoys, and spots defined in `regions.yaml`; adding a new region requires no code changes

---

## Data Sources

All data is fetched from free or free-tier public services:

- **NOAA NDBC** — live buoy observations and spectral swell data (updated every 30 min); yearly stdmet archives back to 2021 used for NDBC backfill
- **Copernicus Marine (CMEMS)** — ECMWF WAM ANFC with swell partitions (VHM0_SW1/SW2, VTM01_SW1/SW2, VMDR_SW1/SW2), the EURO source for both the dashboard and CSC2. Live fetch via `copernicusmarine` (free `copernicusmarine login` credential at `~/.copernicusmarine/.copernicusmarine-credentials`)
- **Google Earth Engine — `COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H`** — cycle-preserving archive of the same CMEMS product, used by CSC2's historical backfill because CMEMS itself overwrites past cycles. Free for noncommercial use (Community Tier: 150 EECU-hours/month)
- **AWS Open Data — `s3://noaa-gfs-bdp-pds/`** — NOAA GFS-Wave GRIB2 archive with swell partitions back to 2021-04 used by CSC2's historical backfill; byte-range fetches via `.idx` sidecars
- **Open-Meteo Marine API** — live GFS-Wave partition forecasts for the dashboard and live logger (GFS stream only; EURO migrated off Open-Meteo in v1.5)
- **Open-Meteo Forecast API** — ECMWF IFS and GFS wind model forecasts
- **NOAA CO-OPS** — harmonic tide predictions per spot with Surfline-calibrated time corrections

---

## CSC2 — Colesurfs Correction v2 (lead-time-aware forecast correction)

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

**Model (planned).** Two tiers reported side-by-side on the eval page:

- *CSC2 (baseline)* — per-[buoy × lead-hour × variable] linear bias correction. Simple floor; no learned interactions.
- *CSC2 (model)* — gradient-boosted trees (LightGBM) over features: raw EURO partitions, raw GFS partitions, EURO–GFS deltas per lead, sin/cos day-of-year, buoy identity, and **lead hour as a first-class feature** (since the archive preserves full 0..+240 h lead structure for every cycle). One model per output variable.

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
~3 months of paired cycles; target is 24 months so the model can learn
seasonal pattern changes. The GEE backfill brings us to ~12 months of
lead-resolved CMEMS coverage at kickoff; GFS and NDBC backfills cover
the same window with full lead structure; live loggers top both up daily.

**West-coast track** uses identical architecture; models train silently,
artifacts land in `.csc2_models/west/`, and nothing is surfaced on the main
dashboard until explicitly promoted.

---

## Swell Categorization

The app uses a **7-category 2D lookup system** based on both wave height (ft) and period (s), capturing how a given wave height reads very differently at 6 s vs 16 s.

| Category | Description |
|---|---|
| **FLAT** | Wind chop / no surfable energy |
| **WEAK** | Small, marginal |
| **FUN** | Solid fun-sized surf |
| **SOLID** | Well overhead, quality energy |
| **FIRING** | Pumping — big, powerful surf |
| **HECTIC** | Maxing out — dangerously large |
| **MONSTRO** | XXL / tow-in territory |

Thresholds are defined per period band in `swell-categorization-scheme.toml`.

---

## Wind Condition Rating

In regional mode, each spot is classified by surf-quality based on wind direction relative to the coast, sustained speed, and effective gust. The 6 tiers are evaluated hierarchically (first match wins):

| Rating | Color | Condition |
|---|---|---|
| **Glassy** | Green | Offshore, effective gust < 10 mph |
| **Groomed** | Green | Offshore, sustained > 20 mph |
| **Clean** | Green | Offshore (any speed), or any direction with sustained < 5 mph |
| **Textured** | Gold | Sideshore + gust < 15 mph, or onshore + gust < 10 mph |
| **Messy** | Blue | Sideshore + gust < 25 mph, or onshore + gust < 20 mph |
| **Blown Out** | Grey | Everything else |

Wind direction zones are defined relative to each spot's measured shore normal: offshore (≤ 32° from offshore direction), sideshore (32°–115°), onshore (> 115°). Effective gust is capped at sustained × 1.5 to normalize ECMWF's over-reporting.

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

- No wave breaking / beach angle correction — offshore buoy and model data only
- When all three partition streams from a wave source are empty for a given hour, the dashboard displays no swell rather than falling back to combined sea (v1.5 change — honest empty over faked values)
- 46268 Topanga Nearshore sits ~200 m from the coast; the nearest 0.083° CMEMS grid cell is masked as land, so CSC2 skips that buoy's EURO stream until a seaward sample-point shim is added
- The wind particle field is drawn at the model grid resolution (144 points at 4° spacing), not interpolated to a finer mesh
- Tide corrections were calibrated on a single date (2026-04-01) against Surfline and are fixed constants

---

## Changelog

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

### v1.3.3
- **Fix:** Mobile slider no longer jumps on tap — anchor is taken from the current handle position rather than the tap location, so the chart only moves when the finger actually slides

### v1.3.2
- **Fix:** Long Island buoy updated to active buoy 44025 (44017 was decommissioned Feb 2023)
- **Fix:** Buoy "Now" reading now matches the most recent buoy history point — `_parse()` prefers rows with both valid WVHT and DPD (matching the history chart's energy filter)

### v1.3.1
- **Scroll preservation on model switch** — switching between EURO and GFS now keeps the same date/time column at the left edge of the visible table area; previously the table would jump to a different date because column widths differ between models

### v1.3
- **Batched wave forecast API** — all spot forecasts fetched in a single multi-location Open-Meteo call (7 calls → 1); cold-cache load time dropped from ~14 s to ~2–3 s
- **Server-side sunrise/sunset** — computed via `astral` on the backend; eliminates the browser-side Open-Meteo dependency for day/night column shading
- **Parallel backend fetches** — `ThreadPoolExecutor` for buoy and spot wind fetches
- **Response compression** — `flask-compress` adds gzip/brotli on all responses; Cache-Control + stale-while-revalidate headers for Cloudflare edge caching
- **Background cache warming** — server pre-fetches all data every 30 min so user requests always hit warm cache
- **Disk-based cache persistence** — cache survives server restarts
- **Inlined `/api/config`** — config embedded in the HTML template at render time, saving one round-trip on load
- **Batch fallback** — if the batched wave API call fails, falls back to per-spot fetches

### v1.2.5
- **Outage banner auto-clear** — banner is dismissed automatically on a successful `loadAll` so stale outage warnings don't persist
- **Smart-refresh bypass** — manual refresh skips the "no new data" check when an outage banner is visible, allowing immediate retry
- **Outage dismissed flag reset** — `_outageDismissed` resets on each manual refresh
- **Backend: wind cache retry** — `skip_none=True` added to `fetch_spot_wind` and `fetch_spot_wind_forecasts` so failed wind lookups are retried on the next request rather than serving a cached `None`

### v1.2.4
- **Dynamic outage modal** — modal now shows PARTIAL or FULL DATA OUTAGE based on severity, and dynamically lists which data types are affected (Swell/wave, Wind, Region wind, Tide) based on actual failure conditions
- **Structured failure info** — `_showOutageBanner()` accepts structured failure details from `loadAll()`; graceful fallback when called without arguments

### v1.2.3
- **Broadened outage detection** — outage modal now triggers when any data source fails (spot winds, region wind, swell, tides, wind grid, or timeouts) for any model
- **Outage modal → centered popup** — outage indicator converted from a top banner to a centered modal popup
- **GitHub issue auto-check** — outage modal auto-fetches open open-meteo/open-meteo issues and shows count + link
- **Wind API: double-request fix** — `_RATE_LIMITED` sentinel in `fetch_wind_grid` prevents wasteful hourly fallback on 429 responses
- **Wind API: negative cache TTL** — increased from 10 min to 30 min for rate-limited responses
- **Footer layout** — API usage and swell data text consolidated on one line (desktop)

### v1.2.2
- **Outage banner** — fixed amber top banner appears when any upstream data source fails; includes retry button, dismiss button, and link to Open-Meteo issues

### v1.2.1
- **Photography link** — COLE HEINE PHOTOGRAPHY button linking to coleheine.com added to the logo info modal
- **Info modal polish** — refresh button text standardized to all caps; API call count moved to its own line below swell data attribution

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

### v1.1.1
- **Wind forecast fix** — `regionWindData` fetch moved outside the `regionMode` guard in `setModel()`, so switching models while in spot view (or before first entering it) no longer returns stale data from the previous model
- **Version number** — added `v1.1.1` label; displayed in desktop footer and mobile info popup
- **Mobile info modal** — tapping the header logo on mobile opens a popup with a "Refresh Model Now" button, live API usage/swell data tag, and version string
- **Loading screen** — content shifted slightly north of center (`padding-bottom: 18%`) for better visual balance
