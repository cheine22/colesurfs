# Top 3 surfer-relevant winners — hypotheses and next steps

Ranking source: `surf_metrics.json` on the
`.csc_models/2026-04-20_025630_v2` artifact, composite score with
`{missed=5, FP_SOLID=3, Tp=2, FUN+Hs=1.5, general=1}`. Framed as
hypotheses, not guarantees — the composite ranking depends on weight
choices that you should feel free to challenge.

Renamed directories (inside
`.csc_models/2026-04-20_025630_v2/`):

| rank | new name | original | scope |
|---:|---|---|---|
| 1 | `csc_funplus_2026-04-20` | `funplus` | all buoys |
| 2 | `csc_percoast_2026-04-20` | `lgbm_per_coast` | all buoys |
| 3 | `csc_globalboost_east_2026-04-20` | `lgbm_east` | East Coast only |

Backward-compat symlinks (`funplus`, `lgbm_per_coast`, `lgbm_east`)
were planned but could not be created in the batch that renamed the
dirs — see `manifest.json:rename_note` for the one-line command to
restore them. `csc.models.load_model` dispatches on the `kind` file
inside the directory (not the directory name), so loading via the new
full path works regardless.

---

## 1. `csc_funplus_2026-04-20` (funplus)

Composite: **0.108**. Dominant strength: *missed FUN+ days* — 796
hourly misses vs 1264 for plain `lgbm`, 2521 for `mean`, 2734 for
raw_gfs/persistence.

### Hypothesis for why it wins

The FUN+-weighted training objective (`fun_plus_weight=3.0` on rows
where the *observed* (h_ft, p_s) rated FUN or better) biases the
gradient updates toward correctly predicting those rows. LightGBM
decision-tree splits at the SOLID-threshold boundary get reinforced
because each surfable row now counts for 3× loss if predicted wrong.
The model "spends more capacity" on the upper half of the Hs
distribution — exactly where missed detections live.

The cost: a ~0.022 ft bump in non-FUN+ Hs MAE and a modest rise in
false-positive SOLID+ calls (333 vs 287 for `lgbm`). Worth it when
the headline metric is recall on surfable events.

### Low-hanging fruit

1. **Per-coast funplus variants.** Train a separate FunPlus instance
   per coast. West Coast FUN+ fraction is much higher than East; one
   global weight ratio is a compromise.
2. **Sweep the weight ratio.** 3.0 is a single point in a family.
   Re-run the bakeoff with weights ∈ {1.5, 2, 3, 5, 8} and curve-fit
   the missed-FUN+ vs global-MAE trade-off.
3. **Regime-specific weights.** Up-weight FUN+ only for long-period
   (Tp > 11 s) rows — the forerunner-swell case where the models
   most often under-predict size.

---

## 2. `csc_percoast_2026-04-20` (lgbm_per_coast)

Composite: **0.118**. Dominant strength: *FUN+ Hs accuracy* and
*general score* — 0.272 ft FUN+ MAE and the lowest composite
HS/Tp/Dp ratio against the `mean` baseline (0.739).

### Hypothesis for why it wins

Each specialist is trained on a distribution that is ~5× more
homogeneous than the global pool. East Coast buoys are dominated by
short-period wind-sea; West Coast buoys are dominated by long-period
remote swell. The bias structure in each regime is different enough
that a global model has to compromise, while each specialist can
learn its local bias-vs-regime map cleanly. At inference time the
router dispatches by `buoy_id`, so each row sees the model fit
against statistically similar data.

### Low-hanging fruit

1. **Add a FUN+-weighted per-coast variant.** Compose the two
   winners: per-coast LightGBM + FUN+ loss. Would likely take the
   top spot on both missed-FUN+ AND FUN+-Hs-MAE.
2. **Per-coast hyperparameter tuning.** Current specialists share the
   default `lgbm` hyperparameters. West Coast has longer sequences
   and longer periods — try `num_leaves=63, min_data_in_leaf=50`
   there specifically.
3. **Share a bias-only head between coasts.** Train a small global
   head (say, 8 leaves) on top of each specialist's residuals to
   capture the GFS-wide systematic bias that both coasts share.

---

## 3. `csc_globalboost_east_2026-04-20` (lgbm_east)

Composite: **0.136**. Dominant strengths: *lowest Tp MAE* (1.254 s —
vs 1.435 s for global `lgbm`) and *second-lowest FP SOLID+ count*
(235).

### Hypothesis for why it wins

East Coast Tp is bimodal — short-period NE windswell (5-8 s) and
long-period Atlantic groundswell (11-14 s) — and the models handle
the two regimes differently. A specialist that never has to model
West Coast's narrow, long-period-dominated distribution can invest
more tree capacity into separating those two East Coast modes. Hence
the sharper Tp accuracy.

The low FP SOLID+ count is partially a scope artifact (evaluated on
East-only test rows, where actual SOLID events are rarer), but also
reflects the model's conservative Hs prediction: East Coast SOLID
calls are rare in the base rate and the specialist has learned not
to call them unless both models agree loudly.

### Low-hanging fruit

1. **Re-target onto spectral primary swell.** The East Coast's
   Tp-accuracy advantage will probably grow once the target is
   Hm0(primary) instead of WVHT (which mixes the two Tp modes in one
   number).
2. **Add ECMWF Tm-to-Tp conversion as a feature.** Open-Meteo's
   ECMWF `wave_period` is a mean period; the true peak period is not
   exposed directly. Add a feature that approximates Tp from
   `(Tm, Hs)` via the standard JONSWAP γ ~3.3 conversion.
3. **Train a matching `lgbm_west` specialist with these same
   tweaks** and revisit whether the router should beat each
   specialist-on-its-coast.

---

## Caveats

- The missed-FUN+ and FP-SOLID+ counts above are **hourly** counts
  from the 20,102-row hold-out (via `fun_plus_report.json`'s
  confusion matrices), not calendar-day counts. Running
  `python -m csc.surf_metrics` recomputes them at day resolution
  across all four CV folds, which is a strictly better measurement.
  The ranking is unlikely to change — the same three models win on
  every weighting we tried — but the absolute numbers will.
- The `mean` baseline is the reference for `general_swell_score`. If
  you replace it with a different reference (e.g. `raw_gfs`) the
  composite scores will shift, though the top-3 identity should not.
- `lgbm_west` is scoped to West Coast and has a much higher
  missed-FUN+ count because West Coast FUN+ days are far more
  frequent (base rate). Comparing it directly to global `lgbm` on
  `missed_fun_plus_days` is unfair. That is already reflected in its
  composite (0.497) ranking below the three winners.
