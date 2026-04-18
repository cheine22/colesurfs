# CLAUDE.md — colesurfs

Guidance for Claude when editing this repo.

## What this is

Single-page Flask app that aggregates surf-forecast data (NOAA NDBC buoys,
NOAA CO-OPS tides, Open-Meteo wave + wind models) and renders it as a swell
table. No build step and no bundler — `templates/index.html` inlines all JS
and CSS in one file.

## Where things live

- `app.py` — Flask routes, `Cache-Control` rules, background cache warmer
- `buoy.py` — NDBC fetch + spectral swell decomposition
- `waves.py`, `wind.py`, `tide.py`, `sun.py` — other data sources
- `cache.py` — TTL cache + disk write-through + API-call counter
- `config.py` — loads `regions.yaml`; defines palette, wind bands, grid
- `regions.yaml` — single source of truth for regions / buoys / spots
- `swell_rules.py` + `swell-categorization-scheme.toml` — swell → color
- `templates/index.html` — the entire frontend, ~4.5k lines
- Deployment specifics (how the app is served, restarted, tunneled) live in
  a local-only `hosting.md` that is intentionally git-ignored. Check the
  working directory for it when deploy-related questions come up.

## Caching architecture (important when debugging staleness)

Data flows through **three** caches; a staleness bug could live in any of
them:

1. **Origin TTL cache** (`cache.py` → `@ttl_cache`) — in-memory dict with
   write-through to `.cache/*.json` on disk. Current per-fetcher TTLs:
   - `fetch_buoy`: 600 s
   - `fetch_buoy_history`: 1800 s
   - Wave / wind / tide fetchers: 3600 s
   - Wind grids: 6 h hard TTL via `@model_aware_cache`, with model-run-based
     early invalidation
2. **CDN edge cache** — controlled by `Cache-Control` headers set in
   `app.py` → `_add_cache_headers`. `/api/buoys` intentionally has no
   `stale-while-revalidate` so reloads pull fresh readings synchronously;
   wave / wind / tide endpoints keep SWR because those data sources update
   on model-run cadences that are slower than a typical page reload.
3. **Browser cache** — `max-age` component of the same `Cache-Control`.

The background cache warmer (`_cache_warmer_loop` in `app.py`) runs every
1800 s and pre-fetches everything; it piggybacks on the TTL cache, so a
fetcher's TTL must be shorter than 1800 s for the warmer to reliably
refresh it.

To force-clear all caches at runtime: `POST /api/refresh` (rate-limited to
1 call per 30 s per IP).

## Conventions

- No build step — `index.html` inlines all JS/CSS. Do not introduce a
  framework or bundler.
- No comments unless the *why* is non-obvious. Existing comments are terse
  and justified; match that tone.
- Prefer editing files in place over introducing new modules.
- Python style: follow whatever the file already uses.
