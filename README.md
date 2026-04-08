# colesurfs · v1.3.1

A surf forecast dashboard for the NJ / NY / New England coast. Pulls live buoy data from NOAA and wave/wind model forecasts from Open-Meteo, then presents everything in one scrollable view: a color-coded swell table synced to an animated wind map, with per-spot tide predictions and wind condition ratings.

Flask backend, vanilla HTML/CSS/JS frontend. No cloud account or API keys required — all data sources (NOAA, Open-Meteo) are free and unauthenticated.

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

All data is fetched from free, unauthenticated public APIs:

- **NOAA NDBC** — live buoy observations and spectral swell data (updated every 30 min)
- **Open-Meteo Marine API** — ECMWF WAM and GFS wave model forecasts (10-day hourly)
- **Open-Meteo Forecast API** — ECMWF IFS and GFS wind model forecasts (matched to the selected wave model)
- **NOAA CO-OPS** — harmonic tide predictions per spot with Surfline-calibrated time corrections

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

## Known Limitations

- No wave breaking / beach angle correction — offshore buoy and model data only
- EURO WAM doesn't always return secondary/tertiary swell partitions under wind-dominated conditions; the app falls back to the combined wave height
- The wind particle field is drawn at the model grid resolution (144 points at 4° spacing), not interpolated to a finer mesh
- Tide corrections were calibrated on a single date (2026-04-01) against Surfline and are fixed constants

---

## Changelog

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
