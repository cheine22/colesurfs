# CSC surfer-relevant leaderboard

- Generated: `2026-04-20T11:29:40.677227+00:00`
- Artifact: `/Users/cphwebserver/Documents/colesurfs/.csc_models/2026-04-20_025630_v2`
- Lower is better on every column; ranking uses the composite score (weighted sum of normalized columns).
- Metric definitions: see `csc/docs/metrics.md`.

| rank | model | scope | missed FUN+ days | false-pos SOLID+ days | Tp MAE (s) | FUN+ Hs MAE (ft) | general score | composite |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `lgbm_east` | east | 193 | 46 | 1.243 | 0.560 | 0.843 | 0.022 |
| 2 | `lgbm_west` | west | 198 | 51 | 1.816 | 0.354 | 0.876 | 0.075 |
| 3 | `funplus` | all | 307 | 101 | 1.459 | 0.471 | 0.854 | 0.249 |
| 4 | `lgbm` | all | 390 | 83 | 1.445 | 0.476 | 0.850 | 0.292 |
| 5 | `lgbm_per_coast` | all | 391 | 97 | 1.453 | 0.484 | 0.855 | 0.321 |
| 6 | `ridge_mos` | all | 438 | 100 | 1.628 | 0.502 | 0.942 | 0.393 |
| 7 | `mean` | all | 598 | 43 | 1.675 | 0.599 | 1.000 | 0.449 |
| 8 | `raw_euro` | all | 577 | 69 | 2.124 | 0.615 | 1.166 | 0.531 |
| 9 | `raw_gfs` | all | 617 | 75 | 1.531 | 0.846 | 1.138 | 0.542 |
| 10 | `persistence` | all | 556 | 168 | 2.964 | 1.846 | 2.517 | 0.942 |

## Metric weights (composite)

| metric | weight |
|---|---:|
| `missed_fun_plus_days` | 5.0 |
| `false_positive_solid_days` | 3.0 |
| `tp_mae_all` | 2.0 |
| `fun_plus_hs_mae_ft` | 1.5 |
| `general_swell_score` | 1.0 |

## Per-buoy breakdown (best-ranked model)

Model `lgbm_east` — counts summed across all folds.

| buoy | n_days | FUN+ days (obs) | SOLID+ days (obs) | missed FUN+ | FP SOLID+ | Tp MAE (s) | FUN+ Hs MAE (ft) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 44013 | 552 | 121 | 58 | 21 | 8 | 1.404 | 0.575 |
| 44065 | 488 | 184 | 51 | 57 | 8 | 1.281 | 0.605 |
| 44091 | 552 | 261 | 87 | 39 | 13 | 1.212 | 0.477 |
| 44097 | 552 | 325 | 164 | 41 | 9 | 1.091 | 0.561 |
| 44098 | 542 | 231 | 126 | 35 | 8 | 1.233 | 0.590 |
