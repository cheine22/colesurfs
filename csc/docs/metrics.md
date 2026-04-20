# Metrics reference

Two metric bundles live in the CSC artifact:

1. `metrics.json` — the **oceanographic verification suite**, computed
   by `csc/evaluate.py:summarize()` and aggregated across CV folds by
   `csc/experiment.py:_aggregate_folds()`. This is the canonical table
   surfaced on the `/csc` dashboard.
2. `surf_metrics.json` — the **surfer-relevant view**, computed by
   `csc/surf_metrics.py:run()`. This is a second, narrower table meant
   to answer: "for a surfer reading the forecast, which model lies to
   me least?"

Both are always "lower is better" on the columns they headline, but
they weight errors very differently. Use (1) for model-skill
comparisons at the level a forecaster or oceanographer cares about;
use (2) for picking which model to ship to end-users.

## Oceanographic suite (`metrics.json`)

Every metric below is cross-referenced in `csc/glossary.py` — the same
text appears in the `/csc` dashboard's "Definitions" panel.

### Continuous — computed for Hs (m) and Tp (s)

| key | symbol | why it's in the suite | how to read |
|---|---|---|---|
| `hs_bias`, `tp_bias` | signed mean error | direction + magnitude of systematic offset | 0 is ideal; +0.3 m = model over-predicts by 0.3 m on average |
| `hs_mae`, `tp_mae`  | MAE | the single number most people quote | lower better; units match target |
| `hs_rmse`, `tp_rmse`| RMSE | penalises big misses more | ECMWF scorecard metric |
| `hs_si`, `tp_si`    | Scatter Index = RMSE / mean(obs) | dimensionless, lets buoys with different sea-states be compared | ECMWF's wave-verif headline |
| `hs_hh`, `tp_hh`    | Hanna–Heinold index = sqrt(Σ(p−o)² / Σ p·o) | normalised RMSE that downweights bias | common in wave-model papers |
| `hs_r`, `tp_r`      | Pearson correlation | tracking quality independent of bias | 1 = perfect tracking |
| `hs_slope`, `tp_slope` | regression slope (obs on pred) | peak under- vs over-amplification | <1 = model too flat; >1 = model too spiky |
| `hs_acc`, `tp_acc`  | Anomaly correlation | correlation of anomalies about climatology | ECMWF scorecard; ~0.6+ is useful |
| `hs_nse`, `tp_nse`  | Nash–Sutcliffe | skill vs a constant-mean null | 1 = perfect; 0 = ties climatology; <0 = worse than climatology |
| `hs_skill_vs_baseline`, `tp_skill_vs_baseline` | 1 − MSE(model)/MSE(raw_GFS) | "did we beat GFS?" | +0.10 = 10% lower squared error than raw GFS |

### Direction

| key | meaning |
|---|---|
| `dp_circ_mae` | shortest-arc angular MAE, degrees |
| `dp_bias` | signed mean via the complex-mean trick |
| `dp_vecr` | vector correlation on unit direction vectors |

### Event skill — POD / FAR / CSI

Computed at Hs thresholds 2 / 3 / 4 / 6 / 8 / 10 ft. Reported as e.g.
`pod_hs_gt_4ft`, `far_hs_gt_4ft`, `csi_hs_gt_4ft`.

- **POD (probability of detection)**: of observed events, what
  fraction did the model call? 0–1, higher better.
- **FAR (false-alarm ratio)**: of events the model called, what
  fraction did not happen? 0–1, lower better.
- **CSI (critical success index)**: hits / (hits + misses + false
  alarms). Combines POD + FAR into one number.

### Aggregation across folds

Each numeric metric has a sibling `<k>_std` showing ±1 SD across the
4 folds. A small `_std` = the model is consistent across seasons.

## Surfer-relevant suite (`surf_metrics.json`)

Lives in `csc/surf_metrics.py`. Five columns, ordered by user-priority
weight. All lower-is-better.

| # | key | definition | why surfers care |
|---|---|---|---|
| 1 | `missed_fun_plus_days` | calendar days where observed peak-Hs rated FUN or better (via `swell_rules.categorize`) but the model's peak-Hs forecast did NOT. One count per (buoy × day). Summed across buoys and folds. | This is the "you told me it'd be flat but I showed up to a FUN day" failure. Heaviest surfer-trust cost. |
| 2 | `false_positive_solid_days` | calendar days where the model called SOLID or better but observation did not. | The "drive 2 hours, it's junk" failure. Second-heaviest. |
| 3 | `tp_mae_all` | Tp MAE (s) on the full held-out set. | Surfers read Tp to pick spots by exposure and refraction regime. Period accuracy matters independent of size. |
| 4 | `fun_plus_hs_mae_ft` | Hs MAE (ft) restricted to rows where the OBSERVED category was FUN+. | The size-accuracy we show in the "should I go out" window. |
| 5 | `general_swell_score` | composite Hs / Tp / Dp MAE normalised to the `mean` baseline (`baseline=1.0` per axis); mean of the three ratios. | Catch-all for "don't sacrifice the boring-day accuracy either". |

**Composite rank score** — min-max normalises each of the five to
[0,1] across models, weighted-sums with `{missed=5, FP_SOLID=3, Tp=2,
FUN+Hs=1.5, general=1}`, divides by total weight. Lower is better.

### Daily reduction rule

For each (buoy, calendar UTC day), reduce the hourly observation /
forecast records to a single "peak-Hs" row — the hour with the
largest Hs that day — and categorise that row via
`swell_rules.categorize(height_ft, period_s)`. Missed / false-pos
counts are then per-day, not per-hour, so a single 6-hour FUN event
counts once, not six times.

### Reference baseline for the general score

The `mean` candidate (simple 50/50 GFS+Euro average) is the
reference. Every model's general_swell_score is read against that; by
construction `mean ≡ 1.0`. A model below 1.0 is generally more
accurate than the raw ensemble average.

### How to regenerate

```
python -m csc.surf_metrics --out .csc_models/current
```

Writes `surf_metrics.json` into the artifact dir and
`csc/docs/model-comparison.md`. Safe to run as often as you like; it
does not touch `metrics.json` or the trained model files.
