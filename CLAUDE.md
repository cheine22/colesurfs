# CLAUDE.md — colesurfs

Guidance for Claude when editing this repo. Everything here describes the
code as it is; the historical record is `README.md` § Changelog.

## What this is

Single-page Flask app (v2.0.0) that aggregates surf-forecast data — NOAA
NDBC buoys, NOAA CO-OPS tides, Copernicus Marine / Open-Meteo wave models,
Open-Meteo point winds — and renders it as a swell table synced to a wind
map whose field comes straight from the GFS and ECMWF IFS GRIB grids. No
build step and no bundler: `templates/index.html` inlines all of the
dashboard's JS and CSS in one file.

Alongside the dashboard, `csc2/` is a forecast-correction model trained on
paired (EURO forecast, GFS forecast, buoy observation) triples to predict
corrected primary + secondary swells. Trained models live in
`.csc2_models/`; `/csc` is the eval page (archive table, registry, metric
tables, live correction row). The dashboard itself only links to `/csc` —
there is no CSC2 toggle on the main table.

Other pages: `/review` (Conditions Reviewer), `/seasons` (Seasonal
Analysis), `/gland` (+ `/gland/tuner`), `/tuner`, `/csc-model`,
`/palette-preview`. The iOS widgets are served from `/widget/*` and
`/api/widget*`.

## Repository map

- `app.py` — Flask routes (listed in its docstring), `_add_cache_headers`,
  rate limits (`/api/*` 240/min per IP; `POST /api/refresh` 1 per 30 s per
  IP), the LAN-only `_restrict_tuner` gate (`/tuner`, `/gland/tuner` and
  their save routes), the background cache warmer, `/api/config` (one
  `_config_payload()` served and inlined into `index.html`: spots, swell
  categories + bands, `wind_rating`, `tile_style = bathy.STYLE`,
  `model_colors`, `wind_spots`, `region_views`), `/api/buoy_historical_context`
  (per-hour observed + model-agreement record, backed by
  `.csc2_data/forecasts/` reads for CSC2 buoys; null agreement elsewhere),
  the widget routes, `/api/csc2/*`, `/api/gland/*`, `/api/review*`,
  `/api/fun_days`. `COLESURFS_DEBUG=1` enables `/api/debug/spectral/<id>`
  and Jinja `TEMPLATES_AUTO_RELOAD`. `COLESURFS_PORT` / `COLESURFS_HOST`
  override the bind (0.0.0.0:5151 so LAN devices reach `/tuner`; public
  traffic arrives via the Cloudflare tunnel → loopback).
- `buoy.py` — NDBC fetch + spectral swell decomposition. `fetch_buoy_history`
  defaults to a 10-day range; each record carries a raw
  `spectrum: [[freq_hz, energy_density_m2/Hz, direction_deg | null], …]`
  field, sourced from the same `.data_spec` + `.swdir` bytes already parsed
  for component decomposition (no extra HTTP). `_spectral_components`
  partitions the WHOLE spectrum (valley watershed; peaks merged when < 35°
  apart, the valley ≥ 70 % of the smaller peak AND within a 1.5× frequency
  ratio — without the ratio guard a small 10 s swell on a 5 s windsea's
  shoulder merges into it), reports each partition's **peak period** and
  **peak-bin direction** (Surfline's conventions — energy-weighted mean
  period / circular-mean direction read ~0.3 s long and 10–15° off), then
  drops partitions under `wave_common.MIN_SWELL_PERIOD_S` (5.0 s, same floor
  as the models) or under 0.2 ft and keeps the top 2 by H²·T. Peak period is
  quantised to NDBC bins, so a reading can hop 5.6 ↔ 5.9 ↔ 6.3 s. stdmet ↔
  spectrum pairing is ONE rule, `_nearest_spectral_key` (≤ 30 min,
  inclusive), used by both `fetch_buoy` (BUOY NOW cell, CSC2 obs logger) and
  `fetch_buoy_history` (modal) — NDBC posts the two files minutes apart.
  Golden test: `development-assets/tests/test_buoy_decomposition.py`. Obs
  shards store partitions only (no raw spectrum).
- `waves.py` — Open-Meteo GFS-Wave partition fetch (EURO lives in CMEMS):
  three swell partitions + the wind-sea partition (`wind_wave_*`);
  `_build_components` is the canonical GFS processor. No combined-sea
  fallback (see Pitfalls); an empty component list is an empty cell.
- `waves_cmems.py` — Copernicus Marine ANFC EURO fetch + shared processing
  pipeline (Tm01 × 1.20, 5 s floor, energy-sorted top-2) over SW1/SW2 + the
  wind-sea partition (`V*_WW`). `wind_sea=False` (keyword on
  `fetch_cmems_point` / `raw_rows_to_hourly_records`, part of the cache key)
  ranks swell only — `/gland` uses it. `raw_rows_to_hourly_records` is the
  canonical entry point for historical EURO sources.
- `wave_common.py` — shared `_safe` / component builder / record schema used
  by both wave modules. `rank_with_wind_sea` ranks the model's wind-sea
  partition as one more candidate (`type: "windsea"`), same 5 s floor and
  H²·T order — see "Wind sea is a ranked candidate". Behaviour locked by
  `development-assets/tests/test_wave_identity.py` (golden fixtures);
  regenerate goldens only for intentional changes via
  `development-assets/tests/regen_golden.py`.
- `wind.py` — per-spot Open-Meteo winds ONLY (table cells, ratings, Fun+
  gate, widgets): `fetch_all_spot_winds` batches every spot's current wind
  into one call (`fetch_spot_wind` per spot is the fallback),
  `fetch_spot_wind_forecasts` (WIND row), `fetch_region_wind_forecasts(model,
  past_days)` (regional mode; `past_days` up to 30 for the history strip),
  `estimate_model_run` / `make_new_run_checker` (run indicator, smart
  refresh). Models `config.WIND_MODELS`: EURO → `ecmwf_ifs`, GFS →
  `gfs_seamless`. The map's grid is `wind_field.py`, never this module.
- `wind_field.py` — the map's gridded wind. See "The map".
- `bathy.py` — self-rendered basemap + coastline tiles. See "The map".
- `tide.py` — CO-OPS harmonic predictions with the per-spot Surfline
  offsets from `regions.yaml`. `_fetch_station_window` pulls a
  month-anchored ~4-month window per station (31 days back, 62–92 forward)
  cached 45 days, sliced locally per request. `_annotate` stamps each
  high/low event's corrected time and height (`hilo_iso`,
  `hilo_height_ft`) — the live widget's chart labels use them.
- `sun.py` — sunrise/sunset computed locally (astral).
- `cache.py` — TTL cache + disk write-through + API-call counter
  (`record_api_calls`). Per-key single-flight locks so concurrent misses
  can't stampede a slow upstream (CMEMS cold fetch ≈ 90 s).
  `model_aware_cache` adds run-hour invalidation (EURO waves).
- `config.py` — loads `regions.yaml` into `SPOTS` (one per buoy region),
  `WIND_SPOTS` (one per surf spot: tide station + offsets, shore normal,
  Surfline URL), `REGION_VIEWS` (map centre/zoom, phone variants); palette,
  model ids, unit helpers, `MODEL_UPDATE_HOURS_UTC` (GFS 4/10/16/22, EURO
  10/21 — CMEMS ANFC publishes the 00Z run ~08:30Z and the 12Z run ~20:50Z;
  upstream is hard-capped at 2 cycles/day), `WIND_UPDATE_HOURS_UTC`.
- `regions.yaml` — single source of truth for regions / buoys / spots (seven
  regions: NY Harbor Entrance, Block Island Sound, Long Island, Cape Cod,
  Massachusetts, Barnegat, NH & North Shore, MA). Adding a region needs no
  code.
- `swell_rules.py` + `swell-categorization-scheme.toml` — swell → category
  (period bands, height thresholds; `always` / `never` / `>=N` markers) and
  the per-category colours (`COLORS`; FLAT dark is `#17171b` / `#5c5c72`,
  lifted so the cell doesn't vanish into the page).
- `wind_rules.py` + `wind-categorization-scheme.toml` — wind → rating
  (sustained-only thresholds; `gust_mph` is accepted and ignored; legacy
  `*_gust_max` keys read as sustained). Both schemes reload live on a tuner
  save or `POST /api/refresh`.
- `fun_days.py` — observed fun+ ledger. See "Observed fun+ ledger".
- `gland.py` + `templates/gland.html` — the `/gland` page (Grajagan, East
  Java). Deliberately NOT wired through `regions.yaml`: there is no NDBC
  buoy and no CO-OPS station within thousands of km, so the dashboard's
  data spine is inapplicable. Its own sources: GFS-Wave via Open-Meteo
  Marine, ECMWF-WAM via `waves_cmems.fetch_cmems_point` (lat/lon-generic),
  Open-Meteo `sea_level_height_msl` for tide, Open-Meteo for wind, AODN/IMOS
  near-real-time wave buoys off Western Australia as upstream sentinels. See
  "G-Land landmarks".
- `gland-swell-categorization.toml` + `templates/gland-tuner.html` — G-Land's
  OWN FLAT..MONSTRO thresholds and the `/gland/tuner` page that edits them.
  A separate file from `swell-categorization-scheme.toml` on purpose: the
  site-wide scheme is tuned for NY/NJ beach breaks, G-Land is a long-period
  Indian Ocean point break. `gland.load_gland_bands()` / `categorize_gland()`
  keep their own cache and never call `swell_rules.load_bands()`, so editing
  one scheme cannot move the other (verified by a 924-cell height × period
  sweep). Only the category *names* and *colours* are shared.
- `gland_euro_archive.py` — rolling 14-day CMEMS EURO archive at G-Land's
  offshore node (`.gland_data/euro_archive.json`, `com.colesurfs.gland-euro`
  every 6 h) so `/gland` history mode has EURO alongside GFS. Pulls its own
  explicit UTC window (not `fetch_cmems_point`, which is pinned to the
  forecast window) but processes through
  `waves_cmems.raw_rows_to_hourly_records`.
- `csc2/` — the CSC2 package. See "CSC2".
- `templates/index.html` — the dashboard, ~6.5 k lines. Script sections are
  numbered (`═══ N.`): 0 loader, 1 state, 3 theme + tile helpers, 4 wind
  field / raster / particles, 5 scheduler, 6 wind target tracking, 7 data
  loading, 8 legend, 9 table, 10 cells, 11 hover sync, 12 sun, 13 swell
  arrows, 14 regional wind markers, 15 coast-angle diagnostic, 16 map, 17
  regional mode, 18 controls, 19 theme, 21 side-by-side, 22 helpers, 23
  startup. See "Dashboard landmarks".
- `templates/review.html`, `templates/seasons.html` — see "Review / Seasons".
- `templates/csc.html` — the CSC2 eval page; `templates/csc-model.html` —
  the training-pipeline documentation page.
- `templates/tuner.html` — threshold editor; `templates/palette-preview.html`
  — static palette comparison, no save action.
- `templates/widget_render.html`, `templates/widget_live_render.html` — the
  widget mockups' CSS verbatim, rendered to PNG. See "Widgets".
- `widget/colesurfs.js` — the Scriptable script, served `no-cache`.
- `favicon.svg` + `favicon-{16,32,192}.png` + `apple-touch-icon*.png` — the
  liquid-glass icon set. Glass layers (edge refraction with chromatic fringe,
  convex sheen, lip highlight + tube caustic, foam frost, rim light, depth
  grade) are pure SVG over the wave photo embedded in `favicon.svg`; the
  PNGs are re-exported from it (32 px drops the fringe/spray, 16 px keeps
  only grade + lip + sheen; apple-touch is square full-bleed since iOS masks
  it). Prior icon sets: `development-assets/old-icons/`. Icons are served
  `no-cache` so a swap can't stick in the edge.
- `interface-guide.png` — annotated production screenshot at repo root,
  maintained like README: regenerate whenever a visible UI change lands
  (headless-Chrome capture of `:5151` at 1440×1026
  `--force-device-scale-factor=2`, composed by
  `development-assets/docs/make_interface_guide.py`; badge coords are
  layout-specific, rebuild them from the fresh capture).
- `development-assets/` — dev-only, gitignored, never synced: golden-fixture
  tests (`tests/`, with `conftest.py`), design mockups (`design-demo/` — the
  widget mockups and the agreement-chip variants live here), doc sources
  (`docs/`: `data-flow.svg`, `gland-cheatsheet.md` — the researched notes
  behind the page's cheat-sheet panel, `make_interface_guide.py`), icon
  archive, local-dev launcher.
- `purgatory/` — shelved loading-screen work, gitignored; don't revive it
  unasked. `_hold/` — staging for files awaiting manual review/deletion.
- `hosting.md` — deployment specifics (how the app is served, restarted,
  tunneled); local-only and gitignored. Check the working directory for it
  when deploy questions come up.
- `requirements.txt` — Flask / Waitress / requests / PyYAML / astral,
  eccodes (GRIB), scipy (coast mask), the CSC2 stack (pandas, pyarrow,
  LightGBM, scikit-learn, duckdb, netCDF4), copernicusmarine + xarray.
  Pillow is NOT a dependency: `bathy.py` has its own TIFF reader and PNG
  writer, `app.py` its own PNG decoder for widget tiles.

## The map (v2.0)

### wind_field.py — the gridded wind

Pulls the models' own 0.25° 10 m u/v straight from GRIB: GFS through
NOMADS' grib filter (a subregion cut of UGRD/VGRD, ~23 KB a step; AWS
`noaa-gfs-bdp-pds` `.idx` byte ranges as the fallback), ECMWF IFS through
the open-data `.index` files + byte ranges (one global field per step,
~0.75 MB; `data.ecmwf.int`, AWS mirror as fallback). Steps: GFS hourly to
120 then 3-hourly to 240; ECMWF 3-hourly to 144 then 6-hourly to 240 (06/18Z
stop at 90). Each run is decoded with eccodes (`_decode` normalises row order
N→S and longitudes to −180..180; `_cut` slices the envelope 30–48° N ×
82–55° W, 73 × 109), quantised to 0.05 m/s int16 (`SCALE` 20) and written to
`.cache/wind_field/<MODEL>/<YYYYMMDDHH>.npz` as soon as its first steps land.
Runs are kept `KEEP_DAYS` = 4 days (the composite only reaches back 3).

`update()` runs every 10 min (`start_updater`, a daemon thread; `_loop`
loads the disk store first) and fetches or tops up the two newest cycles
that should be published (`READY_AFTER`: GFS +3 h 30, EURO +6 h 40; cycles
00/06/12/18Z), stopping at the first unpublished step past +24 h since
sources publish in order (a lone skipped file is tolerated; three misses in
a row stop the pass).

`series()` composites the newest run for every valid hour in [now − 3 d,
newest run + 240 h] — newest-run-wins, hours beyond +48 h and past hours
kept 3-hourly, the first 48 h at the model's cadence — and pre-builds gzip'd
chunks (A: now − 3 h → +48 h, B: the tail, C: the past). Cached until a run
changes or an hour passes (the window and `now_index` move). The series id
is `{model}-{newest run}-{n steps}-{YYYYMMDDHH}-{blake2b tag}`; the tag
hashes which run supplies each step, so a top-up that only re-sources hours
the previous cycle already covered is a new id. Wire format `_encode`: per
field, int16 x-deltas within each row (first column raw), the high bytes of
every value then the low bytes; the client prefix-sums each row (int16
wraparound cancels exactly).

Routes (app.py): `/api/wind_field/meta?model=` (grid, steps with local ISO
+ epoch ms + run + lead, `now_index`, chunk layout, runs; 503 until the
first run lands), `/api/wind_field/data?model=&series=&chunk=k|step=i`
(409 when the series id moved — the client refetches meta; cacheable
`public, max-age=86400` because the id is in the URL, the one `/api/`
exception in `_add_cache_headers`), `/api/wind_field/status`. Nothing here
touches Open-Meteo. `python wind_field.py [GFS|EURO]` runs one pass by hand.

### bathy.py — basemap and coastline tiles

Per tile, raw float32 elevation from NOAA NCEI `DEM_global_mosaic`
(exportImage, no key — NOT `DEM_all`, which is coastal-only and sparse/wrong
at zoom ≤ 6), fetched with `MARGIN` (32 px at the 2× render) of neighbour
context. `STYLE` is `"v6"`; bump it for ANY palette / ramp / mask / line
change — the route sends immutable cache headers and 404s any other style.

Styles: `dark` / `light` (the base: land painted flat from the cleaned
mask, sea shaded by depth — half the ramp across the shelf 0–200 m, half
down the slope to 4 km) and `coast-dark` / `coast-light` (transparent except
an anti-aliased line on the land/sea boundary — black in both themes — plus
a translucent land wash, `COAST[*]["wash"]`, lighter and muted so land
stands apart from the sea), drawn ABOVE the wind colour field so wind shows
over land and the coast stays a clean edge. `?dpr=1|2` picks the variant:
the dpr-2 (retina) coast tile is emitted at the full 2× render, 512 × 512 px,
with no downsample (`COAST_HALF_WIDTH[2]` = 2.8 px at that render), so a
DPR-2 screen is 1:1 and a DPR-3 phone downsamples instead of blurring a
256 px tile up — a crisp ~1 CSS px line; the dpr-1 variant is box-filtered
2× → 1× to 256 px (`COAST_HALF_WIDTH[1]` = 2.6). `COAST_EDGE` keeps the core
crisp. The base tiles are always box-filtered to 256 px. One line weight at
every zoom — the mask cleanup is what keeps zoom ≥ 10 quiet.

`_coast_mask` cleans the raw 0 m contour in METRES before tracing, with
floors so a rule under a pixel is skipped rather than rounded up (Block
Island would otherwise vanish at zoom 7): scipy opening removes land
specks/hairlines < ~40 m (≤ 4 iterations), closing fills creeks < ~120 m
(≤ 6), enclosed ponds < 0.25 km² are filled, islands < 1.5 km² dropped —
`hole_px` / `speck_px` clipped to [4, `MAX_COMPONENT_PX` = 600] so a
component is only ever removed when every neighbouring tile's margin window
sees all of it too (that is what keeps seams clean); components touching the
window edge are kept. Pond filling is by size, not edge contact, so a pond
straddling a seam is treated alike on both sides. The base tile paints land
from the same mask (`_render(elev, theme, mask)`), otherwise filled creeks
ghost through the wind field as dark lines.

One NOAA fetch covers a `BLOCK` × `BLOCK` (2 × 2) group of tiles plus margin
and renders every variant of each (`_render_block`, per-group lock so
Leaflet's parallel first requests don't fetch twice); a failed group falls
back to the single tile asked for; NOAA is tried 3× with backoff (0.8 s ×
attempt). PNGs persist under `.cache/bathy_tiles/<STYLE>/<style>[@<dpr>x]/z/x/y.png`,
zooms 5–13 (13 = the page's maxZoom 12 + detectRetina) inside a NY/NE
envelope (`_in_envelope`, so the route is not an open proxy). The route:
misses (out of range, 404) and failures (502) are sent `Cache-Control:
no-store` so no edge ever caches one; hits are `public, max-age=31536000,
immutable`.

`prewarm_async()` at startup (3 workers): the default view (desktop + phone
framing, z 6–9) and every regional view (its zoom and zoom + 1 for retina)
first, then the whole `CORE` box (−76..−68.5, 38.3..43.6) at zooms 6–11,
once. A fresh install is usable within a minute.

### The page's tile handling (index.html § 3)

- `_tileStyle()` reads `CFG.tile_style` — never hard-code the version; a
  stale literal 404s every tile after a style bump. `_tileUrl()` →
  `/tiles/bathy/<style>/<dark|light>/…`, `_coastUrl()` →
  `/tiles/bathy/<style>/coast-<theme>/…?dpr=<1|2 by L.Browser.retina>`;
  `_tileOpts` = `maxZoom 12, detectRetina`.
- `_tileRetries(layer)` re-requests a tile that errored up to 4× with
  backoff (700 ms × n) and an `r=` cache-buster so the browser can't hand
  back the failed response — Leaflet itself never retries; the outline must
  never be seen to vanish.
- `_prefetchTiles` warms the browser cache on moveend/zoomend (600 ms
  debounce): the ring just outside the viewport at the current zoom (pad
  0.6), the viewport one zoom in, a ring one zoom out (pad 0.5), both
  layers; `fetchPriority` low, four at a time, the queue rebuilt on every
  move (a generation counter abandons the old one), abandoned if > 260 tiles.
  Immutable caching makes the real request free.
- Leaflet attribution is hidden site-wide; NOAA NCEI, NOAA NCEP and ECMWF
  are credited in the info modal's provenance list. CARTO watermarks
  key-less raster tiles ("API KEY REQUIRED"); Esri's gray canvas and terrain
  base were tried and rejected (state names, land relief baked in).

### Frontend wind (index.html §§ 4–6, 8)

Four objects, all fed one field at a time:

- **`WindField`** — loads a model's series from `/api/wind_field/meta` +
  `/data`: the step at `now_index` first (one ~15 KB request → first paint,
  `load()` resolves), then chunks A/B/C in the background, `onUpdate` as
  they land. A generation token (`rec.gen`) lets a superseded stream
  (invalidate / 409 / a second load) exit silently; a 409 or
  `invalidate()` marks the record stale and re-reads meta in 250 ms (not an
  error); real failures retry with backoff `min(60 s × failures, 5 min)`,
  `loading` staying set meanwhile so callers don't pile on. Loaded steps are
  kept when the series id is unchanged. Decodes the byte planes into
  `Int16Array`s; `fieldAt(model, ms)` lerps the two nearest loaded steps
  into `Float32Array` m/s, clamped to the loaded range, and names the pair
  (`pair`) and the exact key so a chunk landing only re-renders when it
  changes the pair; `clamped` + `pending` flag a stand-in field while the
  hovered hour's step is still streaming (the status line says "loading
  wind…"). `state(model)` → idle / loading / ok / error.
- **`WindRaster`** — an `L.Layer` whose canvas lives in the custom
  `windRaster` pane (z-index 350, created INSIDE leaflet-rotate's
  `rotatePane`, which is a stacking context — a pane created on `mapPane`
  sorts under every tile). Paints speed per screen pixel: one latitude per
  canvas row, one longitude per column (bearing is always 0), bilinear in
  the grid, `RES` = 2 CSS px per sample, viewport + `PAD` 25 % margin,
  positioned in layer space so it pans with the tiles. 512-entry LUT to
  40 m/s over `WIND_RAMP[theme]` (m/s stops, the only ramp; the legend is
  rebuilt from it per theme): the site's own Obsidian palette made
  continuous — page navy (slate blue in light mode) → indigo → `--accent`
  purple → green → amber → red, lightening past gale force — at `_alpha`
  0.78 dark / 0.8 light. A Windy-style rainbow and muted/saturated variants
  were tried and rejected: the map must stay in the site's palette. Events:
  `zoomanim` scales it like an image
  overlay (`_latLngBoundsToNewLayerBounds`), `zoom` keeps it fitted through
  a pinch (leaflet-rotate drives fractional zooms with no zoomanim),
  zoomend / moveend / resize repaint; `setField` of the same field is a
  no-op.
- **`WindRenderer`** — the particle canvas appended to `overlayPane`
  (viewport + 25 % margin), restarted by `_resubmitWind` on zoomend /
  moveend / theme / resize. Dark strokes in both themes (`rgba(26,26,36,…)`
  dark, `rgba(14,14,30,…)` light), `VEL_SCALE` 0.012 (scaled by viewport
  size and ∛dpr), `MAX_AGE` 90, `TRAIL_LEN` 11; trails drawn in batched
  paths — one `stroke()` per trail position, not per segment. `setData` is
  synchronous (stop → clear → resize → init → start); no internal timers.
- **`WindScheduler`** — 30 ms debounce that pushes one field to raster +
  particles + the time label + the status line (`_setWindRunTag`).
  `reset()` + `WindRaster.clear()` on a model switch so the old model's wind
  never sits beside the new table, then `loadWindForecast` (instant once
  loaded) and back to the user's time.

`_requestWind` / `_resubmitWind` track the current target so map events
re-render without recalculating; `syncToNow` / `syncByTime` →
`_syncWindTo(ms)`. Regional mode draws the same grid; labels/dots are the
Open-Meteo spot winds, dots one colour (`SPOT_DOT`). The outage banner does
not watch the map wind; the status line does. The loader counts the wind
field as one of its ten steps.

Panes bottom → top: bathy tiles (tilePane) → `windRaster` (350) → `coast`
tiles (380) → particles / SVG vectors (overlayPane) → markers.

### Legend and controls (§ 8, § 16)

The legend is one Leaflet control (`LegendControl`, bottomright) added
BEFORE `RecenterControl` (Leaflet inserts later bottom controls above
earlier ones), so it sits directly under `⊙ RECENTER` and wears the
button's clothes: `background var(--bg1)`, `1px solid var(--border1)`,
5 px radius, the same shadow; 150 px wide (118 px on phones). It holds the
displayed time (`#map-time-overlay`, set by `_setWindTimeLabel` — NOW or the
hovered column; there is no separate time badge) above `#wss-body`, which
`buildWindScale` fills with the colour bar (0–50 mph from `windRampColor`)
and the mph ticks, once per theme (`dataset.built`). `_setWindRunTag` only
surfaces `#wind-status` — "loading wind…" / "wind unavailable" — beneath
the bar while there is nothing on the map.

`⊙ RECENTER` is offered only off the default view: `_defaultView()` is the
one definition of where the map rests for the overview or the current
region (desktop/phone centre + zoom from `MAP_CENTER*` / `REGION_VIEWS`, or
`_getBoundsCenterZoom` for a region without a view — exactly what
`fitBounds` would do); `recenterMap()` sets it; `_syncRecenter()` on
moveend/zoomend/resize (and after entering/leaving a region) adds
`.is-home` (display none) when the zoom matches and the default centre is
within 6 px of the map's middle. There is no OVERVIEW button:
`enterRegionMode(name)` on the already-active region calls
`exitRegionMode()`, i.e. regional mode is left by clicking the active
region's name again.

## Data & caching

Data flows through these caches; a staleness bug could live in any of them:

1. **Origin TTL cache** (`cache.py` → `@ttl_cache`) — in-memory dict with
   write-through to `.cache/*.json` on disk. Per-fetcher TTLs:
   - `fetch_buoy` 600 s; `fetch_buoy_history`, `fetch_buoy_historical_context`
     1800 s
   - GFS waves, spot winds, region winds, tide predictions 3600 s
   - `fetch_cmems_point` (EURO waves): `@model_aware_cache`, 24 h hard TTL,
     invalidated on `MODEL_UPDATE_HOURS_UTC` (10/21 UTC) — new runs land
     within one 30-min warmer cycle; upstream is hard-capped at 2 cycles/day
   - Tides: month-anchored ~4-month window per station, 45-day TTL
     (`tide._fetch_station_window`), sliced locally per request
   - `fun_days.all_summaries` / `review_payload` 600 s, `season_tables`
     3600 s; `/api/csc2/forecast` 1800 s
   - The map's wind fields are NOT in this cache: `wind_field.py` keeps its
     own run store on disk and an in-memory series (rebuilt on a new run or
     hourly)
2. **CDN edge + browser cache — intentionally DISABLED for data.**
   `_add_cache_headers` sends `no-store` on all `/api/*` GETs (except
   `/api/wind_field/data`, series id in the URL, 1 day), `no-cache` on
   HTML, `/widget/*` and the icons; tiles set their own headers (immutable
   hits, `no-store` misses). An edge policy of `s-maxage` +
   `stale-while-revalidate` + browser `max-age` makes a new model run take
   2–3 reloads to appear (SWR serves stale while revalidating; browser
   max-age then re-serves that stale copy). Do not re-add edge caching
   without solving that. Freshness is bounded by the origin TTL cache alone;
   the warmer keeps origin hits fast.
3. **Frontend in-memory** — `historicalData[buoy_id]`, populated by
   `preloadHistoricalData()` after the initial render; invalidated on
   `refreshAll()` so a manual refresh re-pulls the historical-context
   endpoint in addition to clearing the origin caches.

An installed web app resumes its frozen page on re-open without reloading,
so `index.html` has a foreground-refresh hook (`_softRefresh`, on
`visibilitychange` / `pageshow` with `persisted`): when the app becomes
visible and the last successful `loadAll` is > 60 s old, it runs
`refreshAll({soft: true})` — the full re-fetch/re-render path minus the
rate-limited `POST /api/refresh` origin bust and the "No new model data
available" short-circuit (which the manual refresh button surfaces as a
toast).

The background cache warmer (`_cache_warmer_loop` in `app.py`) runs every
1800 s: GFS waves, CMEMS, Open-Meteo region winds (sequential within the
group — no concurrent-429 risk), buoys and tides concurrently, then the CSC2
forecast per east buoy, then `fun_days` (after the wind group so today's gate
hours come from the fresh EURO region-wind entry). It piggybacks on the TTL
cache, so a fetcher's TTL must be shorter than 1800 s for the warmer to
reliably refresh it. Startup also calls `bathy.prewarm_async()` and
`wind_field.start_updater()`.

To force-clear all caches at runtime: `POST /api/refresh` (rate-limited to
1 call per 30 s per IP). It and tuner saves remove only the cache's own
md5-named files — never the on-disk last-known-good forecast.

Fault tolerance:
- The last-known-good forecast fallback persists to
  `.cache/lkg_forecast.json` (`_stash_lkg` writes through a per-writer
  `mkstemp` temp file, since a second process sharing `.cache/` could
  interleave into a fixed `.tmp` name before `os.replace`) and is served
  with `_status: "stale"` when a live fetch fails.
- Partially populated payloads (`/api/forecast/*`, `/api/wind`) carry
  `_status: "partial"` so the frontend can badge degraded data (badge UI
  pending design approval — see `development-assets/design-demo/index.html`).
- The frontend stashes the last good payload set in
  `sessionStorage['cs_snapshot_v1']` (≤ 6 h) and instant-paints the table
  from it on reload before fresh fetches land.

Local-only data directories (gitignored):
- `.cache/bathy_tiles/<STYLE>/<style>[@<dpr>x]/z/x/y.png` — rendered tiles
- `.cache/wind_field/<MODEL>/<YYYYMMDDHH>.npz` — decoded wind runs
- `.cache/widget_png/` — rendered widget images + tiles
- `.csc_data/observations/`, `.csc_data/live_log/observations/` — buoy obs
- `.csc_data/wind_archive/year=Y.parquet` — per-spot hourly ECMWF wind (fun_days gate)
- `.csc_data/fun_days/buoy=<id>/year=Y.parquet` — observed fun+ ledger, one row per day
- `.csc2_data/forecasts/model={EURO,GFS}/buoy=<id>/year=Y/month=M/cycle=*.parquet` — forecast shards
- `.csc2_data/live_eval/<model_name>.parquet` — daily live-eval rows
- `.csc2_data/archive_status_cache.json` — cached `/api/csc2/archive_status` payload
- `.csc2_models/east/`, `.csc2_models/west/` — trained model weights
- `.gland_data/euro_archive.json` — /gland rolling EURO archive

## Observed fun+ ledger (fun_days.py)

Behind the "days since last fun+" line in the Fun+ Days cell, the
regional-view "fun+ days this year" row, `/review` and `/seasons`. Applies
`computeModelOverview`'s rule to buoy readings: 3 h windows, night skipped
(sunrise −30 / sunset +30 via astral), primary swell (spectral partition 1,
stdmet partition 0 fallback) categorized via `swell_rules`, wind-gated per
region via `wind_rules` (≥1 spot Textured-or-better), ≥2 windows per tier;
the day's category is the highest tier with ≥2 windows (SOLID = "≥2 windows
at SOLID or better"). Wind history is the same `ecmwf_ifs` model the wind
cells use, pulled from Open-Meteo's historical-forecast API into
`.csc_data/wind_archive/year=Y.parquet` (back to 2019); ledgers land in
`.csc_data/fun_days/buoy=<id>/year=Y.parquet` (built 2019–2026 for every
buoy; other years on demand when the buoy has obs shards). `/api/fun_days`
serves `all_summaries()` (10-min TTL, warmed): ledger rows plus a live
re-derivation of the trailing 3 days, with today's gate hours taken from the
cached EURO region-wind payload. Cape Cod (44018) has no 2026 data at NDBC;
the UI shows `no obs` rather than a zero tally. Ledger rows also carry the
day's peak reading (max H²·T over every obs in the local day, readings
< `PEAK_MIN_H_FT` = 0.5 ft excluded — the proxy otherwise crowns a 0.2 ft @
27 s noise partition) for `/review`.

**Energy is H²·T everywhere** (buoy.py, wave_common, fun_days peak, buoy
modal, /review, /gland) — the wave-power convention Surfline uses.

`droughts(fun_dates)` is the shared drought rule (non-fun+ days between
consecutive runs of fun+ days): `summary()` reports the trailing-365-day
longest gap vs the open gap since the last fun+ day (`drought`,
`current_is_longest`), and `season_tables()` adds `mean_drought` +
`median_drought` per season-year, where the gap set ALSO includes the
season's edge runs (first observed day → first fun+ day, last fun+ day →
last observed day; a season with no fun+ day is one drought) so a straddling
drought lands in both seasons. `review.html` re-implements the same rule
(edge runs included) as `droughtsOf` for the gap histogram — keep the two in
step. `droughts()` itself stays closed-gaps-only for the dashboard's
trailing-365-day longest-drought line.

Daily job `com.colesurfs.fun-days` (04:20 local): `python fun_days.py
--topup --rebuild` — 7-day wind-archive top-up, then full recompute of this
year + last for every dashboard buoy; older years are static once built
(`--rebuild --year YYYY` to redo one).

## CSC2

- `csc2/schema.py` — buoy scope (5 east + 3 west), path layout, forecast-row
  columns. Every csc2 module imports `BUOYS` / paths from here. The
  `CSC2_DATA_DIR` env var redirects the whole tree, so a backfill can build a
  parallel archive without touching the live one.
- `csc2/logger.py` — live forecast logger (`com.colesurfs.csc2-logger`,
  3 AM + 3 PM ET). Pulls CMEMS + GFS via `waves_cmems.fetch_cmems_point` /
  `waves.fetch_wave_forecast` and writes per-cycle parquet shards. **Cycle
  ids:** GFS is tagged by the clock (`_cycle_id`, 00Z at the 07Z capture,
  12Z at 19Z — correct, Open-Meteo has each run within ~5 h). EURO is tagged
  from the data (`euro_cycle_id`): CMEMS ANFC ships two bulletins a day, the
  00Z run at ~08:30Z (reaches D+10 00Z, 240 h) and the 12Z run at ~20:50Z
  (also reaches D+10 00Z, 228 h), so the 07Z capture holds the PREVIOUS
  day's 12Z run and the 19Z capture the same day's 00Z run. Run day = last
  valid − 10 d; run hour = 12 once past ~20:30Z on that day. The logger
  fetches EURO fresh (`fetch_cmems_point.__wrapped__`, not the dashboard's
  TTL cache, which can hand it a series that predates the newest bulletin),
  asks the CMEMS file listing for the newest bulletin
  (`latest_euro_bulletin`, 120 s cap, clock rule on failure) to settle the
  run hour, and skips a series identical to the buoy's latest shard
  (`_same_series`) instead of logging one bulletin twice. Shards from a
  morning capture may start with negative `lead_hours` (analysis hours of a
  12Z run when the series window began at local midnight) — correct, not a
  label error. The logger still writes the `combined_*` columns alongside
  the raw partitions. (Label is `csc2-logger`, not `csc2-log`: the old
  label's launchd state became unspawnable — persistent EX_CONFIG even after
  re-bootstrap — while identical plist content ran under a new label. Old
  plist kept as `.disabled-poisoned-label`.)
- `csc2/obs_logger.py` — live NDBC observation logger
  (`com.colesurfs.csc2-obs`, every 30 min). Appends to the shared
  `.csc_data/live_log/observations/` tree with dedup on (valid_utc,
  partition), stamping the NDBC observation time. Also logs the dashboard
  buoys outside CSC2 scope (`EXTRA_BUOYS`: 44025, 44018) for `fun_days.py`;
  `archive_status` and the trainers still iterate `BUOYS` only.
- `csc2/train.py` — trainer for both architectures (`--scope`, `--version`,
  `--lead-smoothing` default ±2 h, `--holdout-frac` 0.15, `--force`).
  Asserts the time split (max train cycle < min test cycle) and records
  per-target row counts + the inclusion rule in `meta.json`.
  `_apply_dashboard_fallback_gfs` keeps its name and the `gfs_sw1_source`
  column but only tags rows ("partition" / "missing"); missing rows are
  excluded from training.
- `csc2/predict.py` — inference (`predict_for_cycle`).
- `csc2/registry.py` — model discovery (`list_models` only reads directories
  holding a `meta.json`, so archived generations are invisible) +
  `select_top3` ranking. The #1 slot additionally requires sw1_height skill
  ≥ 0 (`SW1_HEIGHT_SKILL_FLOOR`) — a high composite can't mask a model
  that's worse than raw EURO on primary height (that component carries only
  25 % of the composite).
- `csc2/eval_live.py` — daily live-eval pass (`com.colesurfs.csc2-eval`,
  5 AM ET). For every model under `.csc2_models/east/`, re-runs inference on
  cycles that post-date its training run, compares against obs that have
  since landed, and appends one row per (model, eval_date) to
  `.csc2_data/live_eval/<model_name>.parquet`. The registry's composite skill
  stays training-holdout-based; live-eval is informational. Not yet built:
  surfacing live skill on `/csc`, and a drift watchdog (flag when rolling-30d
  skill drops ~25 % below training-holdout skill).
- `csc2/gee_backfill.py` — historical EURO backfill via Google Earth Engine
  ImageCollection (`COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H`).
  Cycle-preserving archive back to 2025-04-28 (00Z cycles).
- `csc2/aws_gfs_backfill.py` — historical GFS backfill via AWS S3
  (`noaa-gfs-bdp-pds`) with byte-range GRIB2 fetches driven by `.idx`
  sidecars (SWELL/SWPER/SWDIR levels 1–3 + surface WVHGT/WVPER/WVDIR).
  Nearest-gridpoint indexes are resolved once and cached (`_NEAREST_IDX`) —
  per-message `codes_grib_find_nearest` was ~95 % of the wall clock. Row
  `time` must be written as New York local (what `records_to_rows` reads) —
  a UTC string there stamps every row 4–5 h late. The archive holds every
  00Z/12Z cycle from 2025-04-28 with the wind-sea columns: 3 h lead steps
  before 2026-04-21, 1 h steps to f120 after it so the live-logger era keeps
  its hourly rows.
- `csc2/ndbc_backfill.py` — historical buoy-obs backfill from NDBC stdmet
  yearly archives (partition=0 / combined sea only).
- `csc2/ndbc_spectral_backfill.py` — historical buoy spectral decomposition
  (partition=1 / partition=2). Reuses dashboard-identical
  `_spectral_components` from `buoy.py`. Three sources, in fallback order:
  yearly closed (`data/historical/swden,swdir/`), monthly closed
  (`data/swden,swdir/<Mon>/`), realtime (~45 days). Buoys 44091/44097/44098
  are not NDBC-archived (USACE/UCONN/UNH-owned), so realtime is their only
  source — invoke `--realtime` to cover them. Output:
  `.csc_data/observations/buoy=<id>/year=Y/spectral[-YYYY-MM | -realtime].parquet`
  with the same schema as stdmet, populated for partition=1/2 only.
- `csc2/cdip_spectral_backfill.py` — CDIP-sourced spectral backfill for
  west-coast buoys.
- `csc2/archive_status.py` — paired-cycle coverage per buoy with file cache
  (`BUOY_OBS_PARTITIONS = (1, 2)`); backs `/api/csc2/archive_status`.

### Current model set and retrain rules

- East and west each hold `CSC2+baseline_260919_0.85_v8` and
  `CSC2+ML_260919_0.85_v8`, trained on the wind-sea-ranked archive after the
  GFS/EURO re-pull. East baseline is #1 (raw-EURO primary-height MAE 0.58 ft
  on the holdout, baseline 0.54 ft → SW1 height skill +0.07, composite
  0.116); west baseline v8 is #1 on its track.
- Archived generations: `.csc2_models/<scope>/_pre-relabel/` (v1–v5: learned
  lead hours 12 h short on the live half of their EURO rows, before cycles
  were labelled from the data) and `_pre-v1134/` (v6, v7: learned late-stamped
  AWS GFS rows and a swell-only SW1, composites scored on a different
  holdout — the registry would otherwise keep v6 on top). The v7 lesson:
  against full-spectrum buoy partitions, partition 1 is a sub-6 s windsea on
  ~15 % of hours, which the models file under wind sea rather than SW1, so a
  swell-only model ranking can't beat raw EURO on primary height (skill
  ≈ 0 / −0.02); ranking the wind sea as a candidate is what closed that gap.
- Retrain cadence: quarterly via `com.colesurfs.csc2-retrain` (1st of
  Mar/Jun/Sep/Dec at 04:00 local). Wipes the archive-status cache,
  recomputes coverage, then runs `python -m csc2.train --version v1 --force`
  for both scopes; auto-derives YYMMDD + coverage. The quarterly job still
  passes `--version v1`; the date in the name is what orders models, so that
  is harmless, but bump the tag by hand whenever the data rules change.
- Consider an off-cycle retrain when: the top performer's live skill drops
  ≥ 25 % for 3+ consecutive days, east-pool paired-cycle coverage gains
  ≥ 30 days since last train, a new buoy-data source is backfilled (new
  `spectral-*.parquet` under `.csc_data/observations/`), or any processing
  rule changes (partitioning, ranking, cycle labelling) — then re-pull or
  relabel the archive first and park the old generation in a `_pre-*`
  directory.
- Known data edges: ~165 live-only EURO cycles per buoy (mostly 12Z,
  2026-04-21 → 2026-09-18) and all of 46268 have no historical source and
  stay swell-only-ranked with null `ww_*` / `displaced_*` / `sw*_type`. Live
  obs-log rows from 2026-04-24 to 2026-09-08 are ingest-stamped (the logger
  then read a key `fetch_buoy` never returned — up to ~40 min late, half
  snapping to the wrong hour in `train._snap_to_hour_iso`, each observation
  under two timestamps); the backfills' latest-ingest-wins dedupe replaced
  everything they could reach, and rows on NOAA-owned buoys between
  2026-04-24 and the start of the realtime window stay on the old
  pre-full-spectrum convention.

### Nomenclature

- **CSC2** refers to the *training dataset* (full GFS + EURO model runs
  paired with buoy obs), not any model. A "CSC2 model" is always a trained
  instance with a name following the convention below.
- **Model instance name:** `CSC2+{baseline|ML}_{YYMMDD}_{coverage}_v{N}`
  - `baseline` vs `ML` — architecture per README § CSC2 ("CSC2 baseline" =
    per-[buoy × lead-hour × variable] linear bias correction; "CSC2 ML" =
    LightGBM GBT over EURO/GFS/delta features + lead hour + DOY). The
    baseline supports count-weighted lead-hour bias smoothing
    (`--lead-smoothing`, default ±2 h; 0 disables).
  - `YYMMDD` — train date in UTC, sorts lexicographically (e.g. `260919`).
  - `coverage` — fraction of 365 (always 365, not 366) where the
    **east-coast pool has ≥1 paired GFS + EURO + spectral-swell-buoy day**,
    rounded to 0.01. "Spectral-swell-buoy" means partition=1 (primary swell)
    or partition=2 (secondary swell) from the dashboard's spectral
    decomposition (`buoy._spectral_components`); partition=0 (combined sea,
    basic NDBC stdmet) is **not** trainable because it doesn't match the
    dashboard quantity the model is predicting against. Computed as
    `len(histograms.combined_east.paired_by_doy) / 365` from
    `archive_status_cache.json`. The metric counts unique paired calendar
    dates uncollapsed across years, so once we cross into year 2 the value
    can exceed 1.0.
  - `v{N}` — architecture / hyperparameter / data-rule variant. Bump for any
    structural change (different feature set, LightGBM params, baseline
    binning, partition or labelling rules).
- **Examples:** `CSC2+baseline_260919_0.85_v8`, `CSC2+ML_260919_0.85_v8`.
- **Weights land in:** `.csc2_models/east/<full-name>/`. The west track uses
  the identical convention under `.csc2_models/west/<full-name>/` and never
  surfaces on the dashboard until explicitly promoted.

### Wind sea is a ranked candidate

Both wave models file a sea under their *wind-wave* partition for as long as
the local wind is still driving it (wave-age test), so during an onshore gale
every swell partition reads exactly 0 m while the model's own combined sea is
10 ft @ 10 s; the hour the wind drops, the same energy is relabelled swell.
The buoy decomposition has no wind-sea concept — it reports that sea as
partition 1 — so a swell-only model ranking disagrees with the buoy on
exactly the biggest days. The wind-sea partition therefore competes with the
swell partitions under the same rules (5 s floor, H²·T) and carries
`type: "windsea"`; the frontend leads such a line with a wind glyph
(`.ws-glyph`, `_WIND_SEA_GLYPH`). Known cost: an offshore blow's chop can
take the primary slot (e.g. 8 ft @ 7 s NNE at Block Island Sound over a
3.8 ft @ 8 s E swell) — the glyph is what flags it, and the buoy reads the
same thing. Records also carry `wind_sea` (the candidate wherever it ranked)
and `displaced_swell` (the swell it pushed out of the top 2), logged as
`ww_*` / `displaced_*` / `sw{1,2}_type`, so a swell-only ranking is
rebuildable from any shard without a re-pull. **/gland is opted out**
(`wind_sea=False`): its window scoring needs both swell partitions, and the
SE trade windsea would take a top-2 slot.

## G-Land landmarks (gland.py)

- **Swell is sampled offshore, not at the spot** (`SWELL_NODE_LAT/LON` =
  −9.00, 114.25, ~32 km SSW). The Blambangan peninsula shadows the inshore
  model cells: GFS-Wave at the cell the pin snaps to (−8.75, 114.25) reads
  11.3 s from 180° while the next cell south reads 15.3 s from 209° in the
  same hour — the inshore cell has lost the long-period SSW entirely. Both
  models read the offshore node, which matches Surfline's deepwater
  convention and keeps GFS-vs-EURO like-for-like. **Wind and tide still come
  from the spot itself** — only waves move offshore. Do not "fix" the node
  back to the pin.
- **Geography is traced, not invented.** `SECTIONS[*].lat/lon`, `REEF_LINE`,
  `HARBOUR_CHANNEL` and `POINT_TIP` were read off Esri World Imagery by
  pixel→latlng conversion, validated by the trace landing on Surfline's pin
  to 4 decimal places. The point tip is at the **SW**; the reef runs
  **north-east** into Grajagan Bay, so sections order Kongs → Moneytrees →
  Speed Reef → (harbour channel) → Chickens → Tiger Tracks with longitude
  increasing. An earlier schematic had this backwards.
- **`SWELL_BANDS` overlap on purpose.** 165–190 outer / 190–210 Speedies /
  **205**–250 Moneytrees / 250–285 outer — so 205–210° feeds Speedies *and*
  Moneytrees. Consumers must handle multiple matches (`bandsFor()` returns a
  list, not a hit); the dial splits overlapping bands radially via the
  `ring` field so both light up. Keep in sync with `SECTIONS[*].best_dir` —
  the dial legend and the scoring must not disagree.
- **The forecast table prints exact degrees, not compass points** — a
  deliberate departure from the dashboard's `toCard()`. At G-Land the
  section a swell feeds turns on a few degrees (the 205–210° overlap), and a
  22.5°-wide compass point cannot resolve that. Don't "harmonise" it back.
- **The point map carries no live data.** Section markers and cards show
  *preferred* size/direction/period/tide only. Live rating and wind belong
  in the forecast table, not on the map.
- **EURO is fetched swell-only** (`fetch_cmems_point(..., wind_sea=False)`,
  and the same flag in `gland_euro_archive`); verified byte-identical to the
  pre-wind-sea output over 264 timeline hours.
- **Swell-window filtering (`pick_gland_swell`)** — the reason this page is
  its own module. The dashboard's energy-sorted "primary swell" is *wrong*
  at G-Land: in the dry season the largest partition is routinely a 7–8 s SE
  trade windsea that the west-facing point never sees, while the surf is a
  smaller 16 s SSW line. Partitions are scored as `H²·T · window_fit`, with
  a hard 9 s period floor (`MIN_GROUNDSWELL_PERIOD`) and a direction taper
  across `WINDOW_EDGE` (165–285°) peaking over `WINDOW_CORE` (190–250°).
  Both models are ranked the same way so model-vs-model stays
  apples-to-apples.
- **`rank_sections`** — G-Land is five waves, not one, so the page ranks
  the reef rather than rating the spot. Score is
  `(dir 30 + period ≤ 15 + tide 30 + wind 25) × size_fit × ceiling ×
  prestige`. **Size is a multiplicative gate, not an additive term** —
  Speedies on a 3 ft day must be 0, not "a bit off"; an additive version
  gave it 27/100 off good wind and tide alone. `_quality_ceiling` caps the
  whole reef by absolute swell size so a relative winner in marginal surf
  still reads marginal. `prestige` encodes that Moneytrees/Speed Reef are
  world-class and Chickens is, per Surfline, "a slightly lame little left".
- **Per-section wind** — there is no single shore-normal here. The tip faces
  due west and the shoreline swings to NNW as the wave wraps, so each
  section carries its own `offshore` bearing (Kongs wants E, inside wants
  SE; ESE is best overall). `wind_for_section` rates against that.
- **Section scoring is server-side only.** `_build_timeline` merges all
  sources onto one hourly timeline and emits a compact `sec_gfs` /
  `sec_euro` score array per hour in `SECTIONS` order. `gland.html` renders
  those numbers and never recomputes them — a draft that duplicated the
  scoring in JS drifted immediately.
- **Translation layer (`translate_upstream`)** — turns WA buoy readings into
  a G-Land arrival, following Collard/Ardhuin/Chapron (2009) swell tracking:
  back-project each buoy along its great circle at `cg = gT/4π`, triangulate
  the single storm position that explains *every* buoy's direction **and**
  agrees on when it radiated (`_fit_source`, coarse 2° grid then 0.5°
  refine), then forward-project that source to G-Land. Height uses geometric
  spreading only (`_spread_factor`, energy ∝ 1/[α·sin α]); dissipation is
  deliberately unmodelled rather than fudged. Picking each buoy's source
  independently by a storm-belt heuristic and averaging scatters sources by
  3,700 km and is meaningless: **confidence must come from rays actually
  converging** (`bearing_err_deg`, `time_spread_h`), not from how many buoys
  share a period bin; a lone buoy always fits perfectly and means nothing.
  **Unvalidated.** There is no backtest — no archive of past buoy readings
  paired with what G-Land actually did. The only check so far is a
  single-instant comparison against GFS/EURO (2026-07-29): height within
  0.2 ft and period carried through, but **arrival direction 18–23° off the
  models**, wider than the 20°-wide Speedies band — so it cannot be trusted
  to say which section a swell will feed. Sensitivity: 500 km of
  source-position error moves arrival bearing only 6.5° but shifts ETA by
  13 h, so ETA is the fragile output. Treat the whole layer as a cross-check
  on the models, never as a replacement, and don't let the UI wording drift
  toward claiming more.
- **`compare_translation_to_models`** puts that cross-check on the page: for
  each in-window cluster it looks up the timeline row at the hour the swell
  is predicted to land (arrive_epoch is UTC; timeline keys are WIB = UTC+7,
  no DST — the +7 h shift and the round-to-nearest-hour are easy to get
  wrong) and reports signed ΔHs / ΔT / ΔDir against GFS and EURO with an
  agree/close/diverge verdict. Per-cell colours are per-axis while the
  verdict combines them, so a green ΔHs next to a red verdict is correct,
  not a bug. The comparison is legitimate rather than circular **because the
  translation has no wave model anywhere in its chain** — in-situ readings
  plus geometry. If anyone seeds the translation from model data, this panel
  stops meaning anything.
- **`fetch_upstream_model_swell` uses COMBINED sea state on purpose.** A
  waverider reports total Hs/Tp/Dp, so combined is the like-for-like against
  an observation. This is the one place on the page that touches combined
  values, and it is model-vs-buoy; the G-Land forecast table stays on
  primary swell, where model-vs-model belongs. Don't "unify" the two.
- **Upstream buoys are sentinels, not intercepts.** `UPSTREAM_BUOYS` are
  2,300–2,900 km away down the WA coast and NOT on G-Land's swell rays; the
  page says so explicitly. `north_offset` is how far off due north G-Land
  sits from each buoy, and `transit_hours` is a distance scale reference,
  not an ETA. Don't let either get relabelled into a promise about arrival.
- Tide events are parabola-refined (`tide_events`); the Tide card lists the
  day's turn times, orders first and spans the grid below 720 px. The `✈︎`
  glyph is `U+2708` + `U+FE0E` (VS15) so WebKit keeps it monochrome text.
- `/gland` shares the review page's shell (sticky header with crumb, cards,
  the `--gap` rhythm, safe-area footer `← DASHBOARD · MORE`, More modal with
  theme + version). G-Land is a trip destination, not part of the NY/NJ/NE
  forecast: it has no row on the dashboard and is reached from the More
  modal only.

## Widgets

`widget/colesurfs.js` + `/api/widget` — the iOS home-screen widgets
(Scriptable). `api_widget` in app.py is a server-side port of
`computeModelOverview` (the Fun+ Days cell: 3 h stride from now, night
skipped, min(EURO, GFS) category, EURO region-wind gate, ≥ 2 windows per
day) — change the JS rule and this together, they must agree exactly.
`last_update` is the OLDER of the two model runs, labelled "today 12Z" /
"yesterday 0Z" by the run's LOCAL day (a 00Z run is the previous evening
here). The phone holds a two-line loader that `await eval`s
`/widget/colesurfs.js` (served `no-cache`; the script body is one async IIFE
because `eval` parses a classic script, where top-level await is a syntax
error), so edits ship through autopull. Small = 1 region, medium = 2,
large = 4.

**The widget is an image, and the live widget mockup is the spec:**
Scriptable can't load the fonts or draw the glass, so `/widget/image.png`
renders `templates/widget_render.html` (the mockup's CSS verbatim, one
widget at the mockup's own px size — `_WIDGET_DIMS` small 170×170, medium
360×170, large 360×376 — iOS rounds the corners) with headless Chrome
(`_CHROME`, writes the PNG then lingers — the route polls for the file and
kills it) at 3×, cached by content hash under `.cache/widget_png/`;
`/widget/render` serves the HTML for debugging. The script asks for the PNG
at the widget's point size × screen scale (`w`, `h`, `scale`; the template
zooms the mockup layout to that box) and, because Scriptable recompresses
any image a widget LOADS above ~500 k px (a 507² small survives, a 1080×507
medium does not — on-phone slicing can't help), the phone fetches
`/widget/tile.png` tiles (`cols`×`rows` ≤ 8, **≤ 290 px a side**,
`MAX_TILE_PX`) that the server crops from the one full render with
`_png_decode` (own reader; sips ignores a 0 crop offset) and lays them edge
to edge at 1:1. `_widget_render_lock` single-flights the Chrome run and the
decode since a widget's tiles arrive as 8–16 parallel requests. Point sizes
come from a screen-width table; the parameter `size=WxH` pins them and
`calibrate` shows a point ruler to read a phone's real size. Numerals
containing a 0 use Archivo (JetBrains Mono has no plain zero). FLAT's ink is
lifted on both widgets (`_FLAT_DARK_INK` / `_FLAT_LIGHT_INK`). Tap opens the
site in Safari: a home-screen web app has no URL scheme and Shortcuts' Open
App can't target one either (tried 2026-10), so `OPEN_SHORTCUT` stays empty;
the server sends each widget's target as `X-Tap-Url` on every tile.

**Live widget** (parameters `live-lido` / `live-landing` / `live-southampton`,
medium only, `kind=live` on the image/tile routes, `/api/widget/live` →
`_live_payload`, `templates/widget_live_render.html` = the live mockup's CSS
+ chart JS verbatim): left = the region buoy's partitions, tinted by the
primary's category — the category word is the hero line and the swells sit
beneath it in the BUOY NOW cell grammar (every category reads the way FLAT
does; FLAT keeps only the primary; a reading with no period keeps the
reading-as-hero layout) + "Swell trending X" = the worst primary rating
either model forecasts in the window, floored at the buoy ("Swell staying
X" when that is the buoy's own rating); right = the spot's tide now + today's
curve with the CO-OPS highs/lows (`hilo_iso` / `hilo_height_ft`; the payload
rebuilds them from `hilo_time` while an older cached tide dict is live) and
wind, tinted by the better of the two models' current ratings. Window = the
rest of today's daylight; after sunset (or before sunrise) it is the coming
daylight and the pane shows "dawn patrol wind" badges at the sunrise hour
instead of the outlook sentence. Outlook tiers: good = Glassy/Groomed/Clean,
Textured, bad = Messy/Blown Out; not good → first hour either model turns
good ("Wind trending glassy at 4 PM"), bad → first textured hour; good →
first hour either model leaves good ("deteriorating at…" for bad, "trending
textured at…"); else "holding for rest of day" / "holding tomorrow" —
`_live_wind_sentence`, mirrored in the mockup's JS. The wind-rating
lightness scale is mirrored there as the `WIND` map (dark inks one step
brighter than the table's). The live widget's halves lean toward the seam
(20 px outer / 10 px inner inset).

Parameters: empty / `forecast-nyc` (regions in yaml order), `forecast-bi`
(Block Island Sound first; anything unrecognised = default), the three live
spots (`spot=` on `/api/widget/live` and the image routes; any `WIND_SPOTS`
entry with a tide station + shore normal works, the buoy is its
`buoy_region`), `calibrate`; `; size=WxH` appends to any. The More modal's
▣ IOS WIDGETS button (`#widget-modal-overlay`) carries the setup steps, a
copyable loader and this table — keep it in step with the script's header
comment.

## Review / Seasons pages

- `templates/review.html` — `/review`, the Conditions Reviewer: per region,
  a days-per-rating histogram, gap histogram ("Days between fun+ swells",
  weekly bins <7 … 35+, captioned with count, average and median), daily
  peak energy (ft²·s, coloured by the day's rating) and daily peak period
  (with the day's period range behind it), one card each, over a review
  period (calendar year / trailing 365 d / season in progress "(so far)",
  listed first / last completed seasons / custom or season + year back to
  2019). Seasons run equinox to solstice on fixed dates — Mar/Jun/Sep/Dec
  21; winter is named for the year of its Jan–Mar part — via
  `fun_days.season_of` and the page's `seasonRange()`, not calendar months;
  the two definitions must stay in step. Reads `/api/review?start&end` →
  `fun_days.review_payload` (10-min TTL), which builds missing ledger years
  on demand when the buoy has obs shards for that year. Charts are
  hand-drawn canvas (no chart library — the page inlines everything like the
  rest of the site); theme key is the dashboard's `wave-theme` so the choice
  follows the user between pages. Layout runs on one `--gap` custom property
  (18 px / 12 px phone) shared by the wrap padding, panel margins and the
  empty-collapsing `.status` line; footer is `← DASHBOARD · MORE` inside
  safe-area insets, version lives in the More modal. The period's fun+ count
  sits once, top-right of the histogram card. `renderFunDef` writes the fun+
  threshold sentence from `CFG.swell_bands` and the surfable-wind sentence
  from `CFG.wind_rating` (duplicated verbatim in `seasons.html` — change
  both); it is minimized by default behind a "How fun+ days and droughts are
  counted" toggle. Links to the Seasonal Analysis under the descriptor.
- `templates/seasons.html` — `/seasons`, Seasonal Analysis: region picker
  only, four season-by-year tables (Flat / Fun+ / `Solid/Firing` = days
  rated SOLID or FIRING / drought length as `avg (med)` days; each column
  shaded by value with its maximum bold, a muted `n/N d` marker on seasons
  the buoy only partly covered, cards two across below 1400 px) from
  `/api/review/seasons` → `fun_days.season_tables` (1 h TTL; overlays the
  same live tail as `rows_between` so the season in progress matches
  `/review` to today's row; persisted to `.cache/` — delete the md5 file for
  key `fun_days.season_tables:():[]` after changing the row schema or the
  restart serves the old shape for up to an hour). Fall 2019 onward — winter
  2019 would need Dec 2018, which isn't archived; `SEASONS_FIRST_YEAR`. Links
  to `/review?region=` at the bottom and in its More modal.
- Both pages share the shell (54 px header, 46 px on phones, logo + page
  name persistent at 18 px; the region select slides into the header's right
  side / its own phone row once the primary selector scrolls under —
  `initCondensedHeader`, `body.condensed`), the remembered region key
  `cs_review_region`, `?region=<buoy_id>` deep links, the More modal
  placement (lower on phones: 18 vh top padding, safe-area aware) and
  `_review_inline_config()` in app.py (the route inlines
  `swell_rules.load_bands()` / `wind_rules.load_config()`, so neither page
  hard-codes either scheme). Neither shows the full rating-scale legend.
  `overflow-x` must stay on `html` only — on `body` it breaks the sticky
  header. The dashboard reaches them from the More modal (▦ SEASONAL
  ANALYSIS first, then ▤ CONDITIONS REVIEWER) and from the regional view's
  fun+ summary row (two text buttons on their own line under the tally and
  drought text).

## Dashboard landmarks (templates/index.html)

- **Loading** — `_loader` counts ten steps (interface, buoys, EURO, GFS,
  spot winds, region wind, alt wind, tides, table, wind field); the page
  instant-paints from `sessionStorage['cs_snapshot_v1']` when present.
- **Table header** — the top-left cell reads **BUOY/REGION** in overview
  mode and **SPOT** in regional mode. Up to two ranked partitions per cell
  (`.comp-line.p1/.p2`), a wind-sea line led by `.ws-glyph`.
- **Fun+ Days column** — `computeModelOverview(spotName)` samples both
  models on a fixed 3-hour stride regardless of UI resolution; tracks best
  `min(GFS_cat, EURO_cat)` for the cell colour and counts days with ≥ 2
  daytime windows ≥ FUN. A window additionally passes a region-wind gate:
  ≥ 1 spot in the buoy's region must rate Textured-or-better (per
  `regionWindData` + `windCondition`) at that hour; the gate is skipped
  ("honest-empty") while region wind hasn't loaded. Denominator = span of
  sampled future times in days. Cell rendered between `spot-cell` and
  `buoy-col` with class `model-overview`.
- **Observed fun+ figures** — `funDaysData[buoy_id]` (from `/api/fun_days`,
  loaded by `loadFunDays()` off the loader's critical path). The Fun+ Days
  cell's second line is `_funSinceHtml()` (`-34 days`, `-250+ days`, blank
  when today already qualifies; hover for date and category); regional view
  appends a `tr.fun-year-row` whose `.fun-year-inner` (`_funYearHtml()`:
  tally, per-category pills, coverage, the drought sentence, the two review
  links) is `position: sticky; left: 0` with `max-width` set from the
  scroller's `clientWidth`. `_applyFunDays()` patches both IN PLACE (called
  from `_afterBuildTable`, the resize handler, and on fetch) — the data must
  never trigger a table rebuild, which would jog the scroll anchor.
- **Region clean-wind** — `_regionCleanWind(region, data=regionWindData)`
  computes, per region per hour, whether ≥ 1 spot rates Glassy/Groomed/Clean
  (`clean`) or Textured-or-better (`surfable`, the Fun+ Days gate) for
  whichever wind model's data is passed. `_windHatchState(region, t, data)`
  returns tri-state `'solid'` (≥ 1 clean) / `'hatched'` (known, none clean)
  / `null` (no wind record — fetch gap or hour outside the wind window).
  Drives the white wind agreement chip, evaluated against BOTH
  `regionWindData` (active) and `regionWindAlt` (hidden model).
  `_cleanWindCache` is a WeakMap keyed on the wind-data object, so each
  model's data caches independently and model switches / refreshes /
  snapshot loads invalidate automatically. `setShowHistory` re-fetches both
  models' region_wind with `past_days` in ALL modes (not just Regional) so
  historical hours are covered. The state name `hatched` is vestigial —
  there is no visual hatch overlay.
- **Agreement chips** — `_agreementChips(spotName, t)` renders up to two
  small tinted letter chips (`.agreement-chip`) stacked vertically in a
  forecast cell's top-right corner (`.agreement-chips`, a right-aligned flex
  column; only from `buildWaveCell`, including the "—" below-threshold path
  — not on buoy-now or historical cells). Each chip is its letter in colour
  `col` on a `color-mix(in srgb, col 13%, transparent)` wash.
  **Swell agreement chip ("M", `.swell`):** a SOLID block in the HIDDEN
  (non-active) model's category colour (`_swellAgreementColors` returns
  `[fill, ink]` = that category's text colour + cell background, so the
  letter is knocked out of the fill) — matches the cell's own colour when
  the models agree, reveals the other model's rating when they diverge (a
  peek without a model switch). Shown ONLY for poor-or-better hidden reads
  (WEAK+); a FLAT or below-threshold hidden read, or a missing alt record,
  → no swell chip. The visibility problem is a *contrast* problem, solved by
  the solid fill — a `min(active, hidden) ≥ FUN` gate to suppress WEAK/FUN
  splits left ~2 chips per 1 600 spot-hours and was reverted; variants are
  weighed in `development-assets/design-demo/agreement-chip.html`.
  **Wind agreement chip ("W", `.wind`):** `col` is neutral-white
  (`var(--text0)`, white in dark mode, near-black in light), shown ONLY when
  BOTH models report ≥ 1 clean spot in the region at that hour
  (`_windHatchState` === `'solid'` for both `regionWindData` AND
  `regionWindAlt`) — a cross-model clean-wind agreement; if either model is
  missing data or reads not-clean → no wind chip. This is why the alt-model
  region-wind fetch (`regionWindAlt`) is loaded alongside `regionWindData`
  everywhere the latter is (re)loaded. Readings are vertically centred
  (`.cell-inner` justify-content: center). The info modal legend
  (`_populateAboutLegends`) renders the same M/W chips.
  **No-overlap:** forecast cells carry a `wave-cell` class;
  `td.cell.wave-cell .cell-inner` gets a right gutter (`padding-right` 13 px
  desktop / 12 px mobile) so the swell reading never runs under the chips;
  spacing inside a reading is `.comp-line { gap: 3px }`, no `&nbsp;`.
  Rejected predecessors: a full-width `.model-agree` text pill (landed on
  ~88 % of cells, buried the reading), filled colour dots, cross-model hatch
  *concordance* (both agree either way), active-model-only clean wind.
- **Wind rating colours** — `windCondColor(cond)` is a LIGHTNESS scale, not
  hues: four tiers of surface lightness from Glassy/Groomed/Clean (light) to
  Blown Out (dark), the ink nearly constant (it flips to light only on light
  mode's darkest tier) so adjacent cells read as one quiet scale beside the
  coloured swell cells; the good tier takes a wider step than the rest so it
  reads as the exception, the lower three stay evenly spaced. No bold on
  good-wind cells. Dark mode keeps the order with the surface sinking below
  the page background and the ink fading toward it. `buildWindCell` sets
  `--wc-text` / `--wc-sub` / `--wc-tide` on the td so `.wind-line` /
  `.wind-gust` / `.wind-tide` take the rating's ink (the tide line is the
  accent mixed 55 % toward that ink). Mirrored in `widget_live_render.html`
  (`WIND`), the tuner's `wind_colors` in app.py and `/gland`'s wind row
  (`windColors` / `_WIND_TIER` in gland.html); the map's speed ramp is
  a different thing entirely.
- **More modal** — `#about-modal-overlay`, opened by the `MORE` button
  (footer on desktop, bottom bar on mobile; the footer carries no other
  buttons). Two `.more-group` blocks: pages (▦ SEASONAL ANALYSIS,
  ▤ CONDITIONS REVIEWER, ✈︎ G-LAND, ∿ CSC (BETA)) above settings (↺ REFRESH
  MODEL NOW, `#theme-btn`, `#sbs-btn` desktop-only, `#pref-show-history`
  button mobile-only, ⇞ TUNER, ▣ IOS WIDGETS). Labels are actions: theme
  reads "switch to <other> mode" (`_themeLabel`), history reads show/hide
  (`_syncHistoryBtn`). On phones it centres in the lower part of the screen
  (18 vh top padding, safe-area aware) so the buttons are thumb-reachable.
  The desktop toolbar carries ▤ CONDITIONS REVIEWER; the info modal's
  provenance list credits every source including the map's.
- **Historical strip** — `_buildHistoricalCellsHtml(stationId,
  resolutionHours)` + `buildHistoricalCell(obs, cellTime)`: a 240 h window
  ending now, `✓` overlay when `model_agreement === true`. Cells carry
  `data-time` so the mobile slider's `_sliderTimes` array picks them up
  alongside forecast cells. Toggle state lives in
  `localStorage['cs_show_history']`. `setShowHistory(val)` syncs the desktop
  toolbar pill-switch (`#desktop-history-switch`) and the More modal's
  history button (`#pref-show-history`), then anchors the rebuild on the
  model-overview column to keep its viewport-x stable across toggles.
- **Background preload** — `preloadHistoricalData()` fires
  `/api/buoy_historical_context` for every buoy in `CFG.spots` after initial
  render (idle-callback). `_scheduleHistRebuild()` debounces re-renders as
  data arrives.
- **Mobile slider** — `_colIndexFromPct(pct)` and `_pctFromColIndex(idx)`
  switch between two mappings based on `localStorage['cs_show_history']`:
  history-off keeps the "small buoy slot at pct=0" buoy view; history-on
  maps `pct ∈ [0, 1]` linearly across the full timeline so pct=0 lands on
  the oldest historical cell. `_sliderResetToNow()` is the canonical "snap
  to now" action (called by double-tap and on the first build via the
  `_sliderResetDone` one-shot). Scrubbing is velocity-scaled.
- **Mobile spot column** — `.swell-table` is translated by `--table-tx`
  (the slider) and `td.spot-cell` counter-translated with
  `position: relative`; `th.th-spot` must stay `position: sticky; top: 0`
  with the same counter-translate — vertical sticky survives
  `overflow-x: hidden`, only the left anchor is lost. Demoting it to
  relative lets the SPOT header scroll away.
- **Safari-tab-only mobile layout** — one block at the end of the
  stylesheet, `@media (max-width: 600px) and (display-mode: browser)`. The
  installed web app reports `display-mode: standalone` and must not change,
  so nothing above that block is touched: in a Safari tab the content box
  uses `100svh` (the `100vh` layout ran under the toolbar), the map's
  flex-basis absorbs the whole toolbar difference so the table keeps the web
  app's height (45 % of `100vh − chrome`), and the signature strip drops the
  21 px home-indicator reserve to 12 px.
- **Touch-action lock** — `.table-scroll *` carries
  `touch-action: pan-y !important` so any descendant cell can't initiate a
  horizontal pan. Combined with `overscroll-behavior: contain`,
  `-webkit-overflow-scrolling: auto`, and `transform: translateZ(0)` on
  `td.spot-cell`, this is the mobile-scroll-stability stack.
- **Map double-tap** — Leaflet 1.9 dropped its legacy touch `tap` handler
  and iOS Safari never synthesizes `dblclick`, so the page implements
  touch double-tap-to-zoom itself (§ 16).
- **Buoy modal** — two stacked canvases (`#buoy-popup-canvas-bot` for the
  spectrum on top visually, `#buoy-popup-canvas-top` for the energy history
  below), fed by `/api/buoy_history/<id>?days=3`. `_drawBuoyChartTop`
  renders the energy line (H²·T, ft²·s). `_drawBuoySpectrum` renders the static
  spectrum at the scrubbed time (default x-axis 0–22 s, auto-extends if
  energy/components reach further). `_updateBuoyTimeLabel` and
  `_updateBuoyInfoStrip` keep the date/time label above the charts and the
  swell readout below in sync on both hover (`_attachChartHover`) and scrub
  (`_buoyScrubApply`). Component labels on the spectrum use a four-candidate
  placement loop to avoid overlap.
- **Interface guide** — regenerate `interface-guide.png` after visible UI
  changes (see Repository map).

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
  exactly the way the live dashboard does — no shortcut pulls of raw CMEMS
  or raw GRIB values. `raw_rows_to_hourly_records` in `waves_cmems.py` is
  the canonical entry point for historical EURO sources; `waves.py`
  `_build_components` is the canonical processor for GFS. If CSC2 output
  disagrees with a dashboard cell for the same hour/buoy, something is wrong
  — not a model difference.
- **Energy is H²·T everywhere.** Never reintroduce H·T² on any side.
- **The live widget mockup is the spec.** `widget_render.html` /
  `widget_live_render.html` carry the mockups' CSS verbatim; change the
  mockup in `development-assets/design-demo/` and the template together.
  Widget tiles stay ≤ 290 px a side.
- Buoys in user-facing output use the location label ("NY Harbor
  Entrance"), with the station number in parentheses only for
  disambiguation. Model-vs-model comparisons show the dashboard quantity —
  primary-swell Hs/Tp/Dp — never combined sea.
- Model-publication facts live in `config.py` (`MODEL_UPDATE_HOURS_UTC`,
  `WIND_UPDATE_HOURS_UTC`); the page's run badge reads the true run.
- Keep `README.md`, this file and `interface-guide.png` in step with
  visible changes; append the version to README's Changelog rather than
  rewriting history.

## Launchd jobs on this Mac

Production runs as user-agent plists at `~/Library/LaunchAgents/` — full
setup/troubleshooting detail in the private `hosting.md`:

- `com.colesurfs.server` — Flask + Waitress on :5151 (log `/tmp/colesurfs.log`)
- `com.colesurfs.autopull` — `git pull origin main` every 90 s. Anything
  committed to `main` ships to production within 90 s — never commit or push
  unless asked; template edits need a kickstart since Jinja caches templates
  at boot (unless `COLESURFS_DEBUG=1`)
- `com.cloudflare.cloudflared` — tunnel to `surfreport.coleheine.com`
- `com.colesurfs.csc2-logger` — CSC2 forecast logger @ 3 AM + 3 PM ET
- `com.colesurfs.csc2-obs` — CSC2 observation logger every 30 min
- `com.colesurfs.csc2-eval` — daily live-eval pass @ 5 AM ET
- `com.colesurfs.csc2-retrain` — quarterly retrain (see CSC2)
- `com.colesurfs.fun-days` — observed fun+ ledger, daily 04:20 local
- `com.colesurfs.gland-euro` — /gland EURO archive top-up every 6 h
  (`python -m gland_euro_archive`)

To reload any service after a code change:
`launchctl kickstart -k gui/$(id -u)/<label>`.

## Pitfalls — don't re-propose

- **Never hard-code the tile style in the page.** It comes from
  `/api/config` `tile_style` (= `bathy.STYLE`); a stale literal 404s every
  tile after a bump. Bump `STYLE` for any palette / ramp / mask / line change.
- **Never reintroduce a period cutoff BEFORE spectral partitioning.** A
  pre-cut deletes 5–6 s primaries outright (the standard summer SE windswell
  at NY Harbor Entrance) and promotes a long-period remnant (0.9 ft @ 9.3 s
  where Surfline read 2.2 ft @ 6 s). Partition the whole spectrum, then
  filter at `MIN_SWELL_PERIOD_S`.
- **Direction gates for wind sea were simulated and rejected.** A
  shore-normal cone is a no-op for regions with east/north-facing spots and
  blocks a due-E gale sea at NY Harbor Entrance; a swell-coherence gate lets
  a 0.2 ft trace partition veto an 8 ft sea. Don't re-propose them without
  new evidence.
- **The GFS combined-sea fallback must stay removed.** Open-Meteo serves
  GFS-Wave partitions for the full 10 days (the archive's null-sw1 share is a
  flat 4–5 % at every lead) and never provides `wave_peak_period` for GFS,
  so a synthesized primary is always at the mean period and only ever
  touches hours where every swell partition is 0 m — which it drew as a FUN
  "primary swell". Those hours are pure wind sea, which the wind-sea ranking
  now shows honestly; both models are honest-empty otherwise.
- **Don't seed the observed fun+ ledger from model swell.** Its whole point
  is that it is observed.
- **EURO cycles are labelled from the data**, never from the capture clock —
  the clock rule put every live EURO cycle 12 h late and understated lead
  hours on half the archive. Retrain after any relabel.
- **Open-Meteo dense grids blow the quota.** The map's wind comes from the
  GRIB sources (`wind_field.py`); never add Open-Meteo grid fetches
  (`fetch_wind_grid`, `/api/wind_forecast` and `config.GRID_*` are gone,
  and so are `WIND_BANDS`, the pill legend, `windCurrent` / `windForecast` /
  `windByTime` and the IDW spot particles). `/api/wind` serves spot winds
  only.
- **Don't re-add edge / browser caching of `/api/*`** without solving the
  2–3-reload staleness it causes (see Data & caching).
- **`_add_cache_headers` must not touch `/tiles/`** — misses are
  `no-store`, hits immutable; an edge-cached 404 would blank a tile for a
  year.
- **`.csc2_models` archives go in `_pre-*` directories**, never deleted; the
  registry ignores anything without a `meta.json` at the model directory
  level.
- **`overflow-x` stays on `html`** on the review/seasons pages; on `body` it
  breaks the sticky header. `th.th-spot` stays `position: sticky`.
- **The installed web app's layout must not change** when fixing Safari-tab
  issues — gate on `display-mode: browser`.
- **`/gland` keeps its own offshore node, its own thresholds, exact degrees,
  swell-only EURO and server-side section scoring** (see G-Land landmarks).
  Its upstream translation never takes model input.
- **AWS GFS backfill rows are stamped in New York local**, not UTC strings;
  a UTC string there shifts every row 4–5 h.
- **Don't use `DEM_all`** for bathymetry; it is coastal-only and wrong at
  zoom ≤ 6.
- **Don't launch a CSC2 logger under the label `csc2-log`** — launchd's
  state for it is poisoned on this Mac.
- **Pillow is not a dependency**; don't add it for PNG/TIFF work (own
  readers exist in `bathy.py` and `app.py`).
- **The More modal is the only home for theme / side-by-side / history
  toggles** — the footer carries just `MORE`.
- **Tide corrections are fixed constants** calibrated once (2026-04-01)
  against Surfline; the known limitation stands.
