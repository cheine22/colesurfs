# Candidate models

Every candidate implements the same fit/predict/save/load interface
(see `csc/models.py`) and is saved under its own subdirectory inside
the dated artifact (`.csc_models/<stamp>_v*/<model_name>/`). Loading
dispatches on the `kind` file inside that subdirectory — not the
directory name — so renaming subdirectories is safe.

Each section: what it does, what features it sees, why you would (or
would not) expect it to work, known limitations.

## Baselines

These are not trained. They exist so every headline metric has a
reference point on the leaderboard.

### `raw_gfs` (`csc/models.py:RawGFSBaseline`)

Predict = GFS analysis, verbatim. Features used: none (passthrough).
Baseline for `*_skill_vs_baseline` in `metrics.json`.

### `raw_euro` (`csc/models.py:RawEUROBaseline`)

Predict = ECMWF-WAM analysis, verbatim. Useful foil — on the current
holdout, raw_euro's Hs MAE beats raw_gfs's by ~35 %, but its Tp MAE
is ~50 % worse (Open-Meteo's Tm-to-Tp conversion differs between
models). Reveals that an ensemble needs to weight Hs and Tp
differently per model.

### `mean` (`csc/models.py:MeanBaseline`)

50/50 mean of GFS and Euro Hs/Tp; circular mean of the two
directions. A frustratingly strong baseline — most operational
multi-model systems do not beat a simple mean by a lot.

### `persistence` (`csc/models.py:PersistenceBaseline`)

"Whatever was observed 6 h ago." Requires an observation series at
fit time; at inference it falls back to raw_gfs when no recent
observation exists (which is the inference-time reality, so on the
dashboard it reduces to raw_gfs). A sanity-check more than a
candidate.

## Trained candidates

### `ridge_mos` (`csc/models.py:RidgeMOS`)

Per-target Ridge regression (α=1.0) on the full engineered feature
set — one model each for Hs, Tp, sin(Dp), cos(Dp). Numerics are
median-imputed and standardised; categoricals
(`buoy_id`, `period_band`, `month`) are one-hot encoded.

Why it exists: classical MOS (Model Output Statistics) is the
baseline any wave-bias-correction paper has to beat. If LightGBM
doesn't beat Ridge, we have an interaction-capture problem, not a
data problem.

Known limitation: linear in the features, so it cannot represent the
"GFS is accurate when Tp is short but biased when Tp is long" kind of
conditional effect that dominates here.

### `lgbm` (`csc/models.py:LGBMCSC`)

Global LightGBM regressor — one booster per target, no sample
weights, no per-coast split. Defaults:

- `n_estimators=600`, `learning_rate=0.05`, `num_leaves=31`,
  `min_data_in_leaf=100`, `feature_fraction=0.9`.
- Categorical features passed natively (no one-hot).

Features: everything `csc.features.feature_columns(use_partition_features=True)`
returns — the raw model analyses, their disagreement features, their
interaction terms (Hs×Tp, period×direction), sin/cos(Dp), DOY
sin/cos, plus the three GFS swell-partition variables.

Why expect it to work: the dominant CSC signal is conditional bias in
(model, period-band, season, coast). Trees are the obvious fit. It is
the current `winner` per `csc.train:_pick_winner` (lowest global Hs
MAE).

Known limitation: optimises MAE uniformly — underrepresented-but-
important classes (FUN+ rows) do not get extra attention.

### `funplus` (`csc/funplus.py:LGBMFunPlus`)

Same LightGBM, same features, but each row gets a sample weight. Rows
whose observed (h_ft, p_s) rated FUN or better get `sample_weight=3.0`,
all others get 1.0. Hs and Tp targets use these weights; sin/cos(Dp)
use uniform weights (direction has no categorical analogue here).

Why expect it to work: if the model chronically misses surfable days
(because they're a minority of the training distribution), weighting
up those rows shifts the optimum toward capturing them. On the current
hold-out, FUN+-subset Hs MAE drops from 0.310 ft (lgbm) to 0.286 ft
(funplus), at a cost of 0.022 ft on non-FUN+ rows.

Known limitation: the weight ratio is a global knob — it cannot learn
"weight FUN+ heavily for East Coast summer but not for West Coast
winter". Per-regime weighting is the obvious next iteration.

### `lgbm_east`, `lgbm_west` (`csc/per_coast.py`)

Same LightGBM, trained only on the matching-coast buoys. East Coast
buoys are swell-poor most of the year (short-period wind sea
dominates); West Coast buoys are long-period NPac / SPac swell. The
error regimes are almost disjoint, so per-coast fits can chase
different bias structures without interference.

Evaluation: metrics are reported on the matching-coast test subset
only (`scope="east"/"west"` in `metrics.json`). Cross-coast inference
is possible (nothing stops it) but meaningless.

### `lgbm_per_coast` (`csc/per_coast.py:PerCoastRouter`)

Wraps `lgbm_east` and `lgbm_west`. At inference time it dispatches
each row to its coast's specialist by `buoy_id`. Unknown-coast rows
fall back to `MeanBaseline` with a logged warning.

Why expect it to work: on the current holdout, it has the lowest
global Hs MAE (0.225 ft) and the lowest FUN+ Hs MAE (0.272 ft) of
any candidate — because each half can specialise without being
averaged out by the other.

Known limitation: two boosters, two joblib files per artifact —
roughly 2× disk and 2× load time. Also, it cannot share information
between coasts where it would help (e.g. both coasts share GFS bias
on long-period forerunner swells).

## Model registry

`csc/models.py:all_candidates()` returns the registry. Adding a new
candidate means:

1. Implement the fit/predict/save/load protocol in a new file or
   append to `models.py`.
2. Add to `all_candidates()` and `load_model()`.
3. Add to `csc/experiment.py:_candidates()` and `MODEL_SCOPES`.

The bakeoff picks it up automatically from there.
