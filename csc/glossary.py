"""Terminology glossary shown on the /csc dashboard.

Each entry: term → (1) 1-2 sentence definition, (2) optional category tag
used to group entries in the UI. Keep definitions tight — they are
references, not tutorials.
"""

from __future__ import annotations

# (term, category, definition)
GLOSSARY: list[tuple[str, str, str]] = [
    # ── Wave terminology ───────────────────────────────────────────────
    ("Hs",
     "wave",
     "Significant wave height. The mean height of the largest one-third "
     "of waves in a wave record — the canonical single-number measure of "
     "sea state."),
    ("Tp",
     "wave",
     "Peak wave period. Period of the spectral-energy peak, in seconds. "
     "Longer Tp = more powerful, longer-travelled swell."),
    ("Dp",
     "wave",
     "Peak wave direction — the direction waves are coming FROM at the "
     "spectral peak, in degrees clockwise from true north."),
    ("WVHT",
     "wave",
     "NDBC's field name for Hs in the hourly buoy feed — same quantity, "
     "just the buoy-file header."),
    ("DPD",
     "wave",
     "NDBC's 'dominant wave period' — functionally equivalent to Tp for "
     "this dashboard."),
    ("MWD",
     "wave",
     "NDBC's 'mean wave direction' — the direction averaged across the "
     "spectrum. Close to Dp for uni-modal seas, different for mixed."),
    ("swell partition",
     "wave",
     "A single wave-system component (primary swell, secondary swell, "
     "wind sea) separated out of the full spectrum. CSC v1 trains on "
     "combined Hs; v2 will train per partition."),
    # ── Continuous-error metrics ────────────────────────────────────────
    ("Bias (ME)",
     "metric",
     "Signed mean error: mean(pred − obs). +0.3 ft = model over-predicts "
     "by 0.3 ft on average. Zero is ideal."),
    ("MAE",
     "metric",
     "Mean absolute error: mean(|pred − obs|). Lower is better. Units "
     "match the target (ft for Hs, s for Tp)."),
    ("RMSE",
     "metric",
     "Root mean squared error. Penalises big misses more than small ones. "
     "Standard in the wave-model literature."),
    ("SI (Scatter Index)",
     "metric",
     "RMSE divided by mean observed value. Dimensionless, lets you "
     "compare across buoys with different typical sea states. ECMWF's "
     "headline wave-verification metric."),
    ("HH (Hanna–Heinold)",
     "metric",
     "Normalised RMSE that downweights bias: sqrt(Σ(p−o)² / Σ p·o). "
     "Common in wave-model validation."),
    ("r (correlation)",
     "metric",
     "Pearson correlation between forecast and observation time series. "
     "1 = perfect tracking, 0 = unrelated, <0 = anti-correlated."),
    ("regression slope",
     "metric",
     "OLS slope of observed on predicted. <1 = model under-amplifies "
     "peaks/troughs; >1 = over-amplifies."),
    ("ACC (Anomaly Correlation)",
     "metric",
     "Correlation of forecast anomalies vs observed anomalies about a "
     "climatological mean. ECMWF scorecard metric; ~0.6+ is useful."),
    ("NSE (Nash–Sutcliffe)",
     "metric",
     "Skill vs a constant-mean null: 1 − MSE/σ²_obs. 1 = perfect, 0 = "
     "no better than predicting the mean, <0 = worse."),
    ("skill score",
     "metric",
     "Generic: 1 − MSE(model)/MSE(baseline). 0 = ties baseline, +0.10 "
     "= 10% lower squared error. The dashboard's 'Skill vs GFS' uses "
     "raw GFS as the baseline."),
    # ── Event skill ───────────────────────────────────────────────────
    ("POD",
     "event",
     "Probability of detection — of all observed events above a threshold, "
     "what fraction did the model call? Range 0–1, higher better."),
    ("FAR",
     "event",
     "False-alarm ratio — of all events the model called, what fraction "
     "did not happen? Range 0–1, lower better."),
    ("CSI",
     "event",
     "Critical Success Index — hits / (hits + misses + false alarms). "
     "Range 0–1; combines POD and FAR into a single event-skill score."),
    ("Heidke skill score",
     "event",
     "Categorical skill vs random-chance baseline. 1 = perfect, 0 = no "
     "better than random assignment."),
    # ── Direction ─────────────────────────────────────────────────────
    ("circular MAE",
     "direction",
     "Mean shortest-arc angular distance. A 350° forecast vs 10° obs = "
     "20° error, not 340°."),
    ("vector correlation",
     "direction",
     "Correlation for 2D unit vectors on the compass. The directional "
     "analogue of Pearson r."),
    # ── This dashboard ────────────────────────────────────────────────
    ("CSC",
     "csc",
     "Colesurfs Correction — a trained bias-correction of the GFS and "
     "EURO wave models against NDBC buoy observations."),
    ("analysis-level correction",
     "csc",
     "The CSC v1 scope: predict the corrected CURRENT value at each "
     "hour from the two models' analyses at that hour, then apply the "
     "same correction at every forecast lead. A lead-time-aware v2 is "
     "training in the background."),
    ("lead time",
     "csc",
     "Hours between when a forecast was issued and the forecast's "
     "valid_utc. v1 treats all leads as one bucket; v2 will model "
     "error as a function of lead."),
    ("held-out test set",
     "csc",
     "Last 20% of the training window, separated by time (not random) "
     "so evaluation reflects the walk-forward setting."),
    ("colesurfs category",
     "csc",
     "FLAT / WEAK / FUN / SOLID / FIRING / HECTIC / MONSTRO — surf-quality "
     "class derived from (height_ft, period_s) via "
     "swell-categorization-scheme.toml. SOLID and above is the "
     "'high-consequence' subset."),
]


def to_payload() -> list[dict]:
    return [{"term": t, "category": c, "definition": d}
            for (t, c, d) in GLOSSARY]
