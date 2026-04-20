# CSC strategy — the decisions you need to know before you modify

Brief, opinionated notes on choices that are not obvious from the
code. If you disagree with one of these, read it carefully before
changing it — each one has a reason.

## 1. Train against combined WVHT (known v1 limitation)

NDBC's WVHT is the significant wave height of the full spectrum, not
of any single swell partition. The user-facing dashboard shows
primary-swell Hs (the largest energy peak, from `buoy.py`'s spectral
decomposition). CSC v1 is therefore calibrated against the combined
sea and its predictions are displayed on primary-swell panels via
`predict.py`'s lookup shim. This mixes conventions.

**Why we did it anyway**: the NDBC/CDIP historical archives expose
WVHT continuously but not the per-partition Hm0 — reconstructing
per-partition history requires spectral files that are only well-
served by CDIP for CDIP buoys. Building a full spectral-Hm0(primary)
training target is a real engineering lift; v1 ships without it to
let us iterate on architecture and feature selection.

**v2 plan**: backfill CDIP spectral energy densities for every buoy
that exposes them, derive Hm0(primary) per hour, re-train on that.
Accept that East Coast NDBC buoys without spectral files either get
dropped or get an approximate spectral reconstruction from the NOAA
`.spec` endpoint.

## 2. 4-fold month-based CV, season-balanced

**Decision**: `csc/cv.py:kfold_month_split()` assigns whole months to
folds, round-robin within each meteorological season, so every fold's
test set contains at least one month of DJF, MAM, JJA, and SON.

**Why we moved off the single 80/20 time-holdout**: on the current
archive, the chronological last 20 % fell almost entirely in fall/
winter — the holdout was 0 % summer and 0.1 % spring. A model that
overfits to fall/winter regimes would score well there. CV restores a
fair picture of generalisation.

**Why not leave-one-buoy-out**: would be nice for out-of-sample-
station evaluation, but 8 buoys × 4 seasons is too few buckets to make
that meaningful, and buoys within a coast are strongly correlated
anyway. We chose season-balance over station-balance for v1.

## 3. LightGBM over deep learning

**Why**: effective sample count after temporal decorrelation is
small. Hourly autocorrelation of Hs is 0.9+ at 6 h, so ~100k rows do
not give you ~100k independent training examples — closer to ~5-10k.
Trees dominate deep learning in that regime.

**Corollary**: don't invest in Transformer or TCN architectures until
we have actually-indep examples (either by subsampling to decorrelate,
or by going to per-spot spectral features where per-hour variance is
higher). Both will come with the v2 partition-level retarget.

## 4. Keep every trained version forever; never auto-promote

**Decision**: `csc/train.py` never flips `.csc_models/current`. That
requires a manual `python -m csc.promote <dir>` per
`csc/promote.py`.

**Why**: the moment we auto-promote, a bad retrain silently ships to
users. Automatic regression checks (skill-vs-previous) are also
seductive but they implicitly assume the regression test is
representative — see point (2). Manual promotion forces the human in
the loop to read the bakeoff report and the FUN+ report, and decide.

Old versions stay on disk forever (none are ever deleted). That is
how `csc.serve.list_trained_versions()` and the `/csc` comparison
dashboard work — they enumerate every preserved version and let the
user diff them against each other.

Disk cost: each artifact is ~20-30 MB (mostly the joblib pickles).
Acceptable for now. Set a cleanup policy when we hit 5 GB.

## 5. Analysis-level correction, not lead-aware

CSC v1 learns `correction(state) = f(current_model_analyses, features)`
and applies the same correction at every forecast lead. This is the
cheapest possible formulation: one model, no lead dimension in the
features, no lead dimension in the target.

**Why this is defensible**: the dominant bias in both GFS and WAM at
these buoys is a smooth conditional bias in (model, period-band,
season, coast) — present at lead=0 and barely changing out to
lead=120h. Fixing that is worth most of the potential win.

**Why this is provisional**: lead-time does matter for the tails —
long-period forerunner swells have real lead-time-dependent error
growth. The forward logger is already archiving the multi-lead axis
(`csc/schema.py:PREV_DAY_LEADS`), so v2 can consume it without a data
migration.

## 6. Serve cache is hot-reloadable on symlink mtime

`csc/predict.py` caches the loaded model and only reloads when the
`.csc_models/current` symlink's mtime changes. Promotion = `os.symlink
+ os.replace` in `csc.promote`, which updates mtime atomically. The
Flask process never needs to restart to pick up a new model.

**Implication**: do not write to the artifact directory in place —
always write a new `<dated>_v*` dir, then atomically re-point
`current`. `csc.train` enforces this by construction.

## 7. Per-coast specialists are scoped during evaluation

`lgbm_east` / `lgbm_west` are trained on their coast's rows only and
evaluated on their coast's test rows only (`scope="east"/"west"` in
`metrics.json`). Their Hs MAE is therefore not directly comparable to
global-scope candidates.

**Why**: if you score a West-Coast specialist on an East-Coast test
fold, you are measuring out-of-distribution extrapolation, which is a
different question. The scope field in `metrics.json` makes the
comparison explicit.

**`lgbm_per_coast` (the router) has `scope="all"`** because it covers
every row — it just dispatches each to the right specialist.

## 8. Two parallel metric tables, deliberately

`metrics.json` is the oceanography view. `surf_metrics.json` is the
surfer view. They disagree intentionally:

- On `metrics.json`, `lgbm_per_coast` tops Hs MAE and everyone is
  within a noise band.
- On `surf_metrics.json`, `funplus` wins by a wide margin on
  `missed_fun_plus_days` even though its Hs MAE is worse.

**Which one do we ship?** Depends on the downstream. The dashboard's
current-winner selector (`csc.train:_pick_winner`) reads
`metrics.json` because that's the cheapest, least-opinionated choice.
A future "user-facing winner" selector should read
`surf_metrics.json` and use its composite score. Do not conflate the
two.

## 9. The `named_winners` map in `manifest.json`

`manifest.json` now carries a `named_winners` map and a
`surf_metrics_top_3` list. These are additive; nothing in the existing
code reads them. They exist so the dashboard (future change) can label
the top 3 in the UI and so `csc.serve.list_trained_versions` could be
extended to preserve identity across retrains without depending on
fragile directory names.

When a new bakeoff runs, `csc.train._pick_winner` still writes the
old `winner`/`winner_dir` fields — the new fields are optional and
can be populated by re-running `csc.surf_metrics` on the artifact.
