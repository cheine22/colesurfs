"""Per-coast CSC variants.

Two specialist LightGBM models — one trained only on East Coast buoys
(44013, 44065, 44091, 44097, 44098), one only on West Coast buoys
(46025, 46221, 46268) — plus a routing wrapper that dispatches each
inference row to the model for its own coast.

Design notes:
  * `LGBMEastCoast` / `LGBMWestCoast` subclass `LGBMCSC` and override
    only `fit()`. Prediction is unchanged — if you hand them a row from
    the other coast they'll happily extrapolate, which is why we report
    their bakeoff metrics on the matching-coast test subset only.
  * `PerCoastRouter` composes both specialists and a mean fallback for
    any row whose buoy_id is outside both coast sets. It persists each
    sub-model as its own subdirectory under the artifact path and
    records `kind=lgbm_per_coast` at the top level so `load_model`
    dispatches correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csc.models import LGBMCSC, TARGET_HS, TARGET_TP, reconstruct_dp


EAST_BUOYS = {"44013", "44065", "44091", "44097", "44098"}
WEST_BUOYS = {"46025", "46221", "46268"}


def coast_of(buoy_id: str) -> str:
    bid = str(buoy_id)
    if bid in EAST_BUOYS:
        return "east"
    if bid in WEST_BUOYS:
        return "west"
    return "unknown"


def coast_mask(buoy_ids: pd.Series, coast: str) -> pd.Series:
    return buoy_ids.astype(str).map(coast_of) == coast


# ─── Specialist LGBMs ─────────────────────────────────────────────────────

class LGBMEastCoast(LGBMCSC):
    name = "lgbm_east"
    coast = "east"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        mask = coast_mask(X["buoy_id"], "east")
        X_sub = X[mask]
        y_sub = y.loc[X_sub.index]
        return super().fit(X_sub, y_sub)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("lgbm_east")
        import joblib
        joblib.dump({
            "params": self.params,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "models": self.models,
        }, path / "model.joblib")

    @classmethod
    def load(cls, path: Path):
        import joblib
        obj = joblib.load(path / "model.joblib")
        m = cls()
        m.params = obj["params"]
        m.feature_cols = obj["feature_cols"]
        m.cat_cols = obj["cat_cols"]
        m.models = obj["models"]
        return m


class LGBMWestCoast(LGBMCSC):
    name = "lgbm_west"
    coast = "west"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        mask = coast_mask(X["buoy_id"], "west")
        X_sub = X[mask]
        y_sub = y.loc[X_sub.index]
        return super().fit(X_sub, y_sub)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("lgbm_west")
        import joblib
        joblib.dump({
            "params": self.params,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "models": self.models,
        }, path / "model.joblib")

    @classmethod
    def load(cls, path: Path):
        import joblib
        obj = joblib.load(path / "model.joblib")
        m = cls()
        m.params = obj["params"]
        m.feature_cols = obj["feature_cols"]
        m.cat_cols = obj["cat_cols"]
        m.models = obj["models"]
        return m


# ─── Router ───────────────────────────────────────────────────────────────


class PerCoastRouter:
    """Routes each inference row to its coast's specialist.

    Unknown-coast rows (buoy_ids outside both sets) fall back to a simple
    mean of GFS+Euro raw predictions so the API doesn't crash; we log a
    warning when that fallback fires. The sub-models are trained by the
    usual `fit(X, y)` call — internally each specialist does its own coast
    filtering.
    """

    name = "lgbm_per_coast"

    def __init__(self):
        self.east = LGBMEastCoast()
        self.west = LGBMWestCoast()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.east.fit(X, y)
        self.west.fit(X, y)
        return self

    def _fallback(self, X: pd.DataFrame) -> pd.DataFrame:
        sin_avg = 0.5 * (np.sin(np.deg2rad(X["gfs_wave_direction"])) +
                         np.sin(np.deg2rad(X["euro_wave_direction"])))
        cos_avg = 0.5 * (np.cos(np.deg2rad(X["gfs_wave_direction"])) +
                         np.cos(np.deg2rad(X["euro_wave_direction"])))
        dp = reconstruct_dp(sin_avg.to_numpy(), cos_avg.to_numpy())
        return pd.DataFrame({
            "pred_hs_m": 0.5 * (X["gfs_wave_height"] + X["euro_wave_height"]),
            "pred_tp_s": 0.5 * (X["gfs_wave_period"] + X["euro_wave_period"]),
            "pred_dp_deg": dp,
        }, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        coasts = X["buoy_id"].astype(str).map(coast_of)
        east_idx = X.index[coasts == "east"]
        west_idx = X.index[coasts == "west"]
        other_idx = X.index[coasts == "unknown"]

        parts: list[pd.DataFrame] = []
        if len(east_idx):
            parts.append(self.east.predict(X.loc[east_idx]))
        if len(west_idx):
            parts.append(self.west.predict(X.loc[west_idx]))
        if len(other_idx):
            buoys = sorted(set(X.loc[other_idx, "buoy_id"].astype(str).tolist()))
            print(f"[per_coast] WARN: fallback for {len(other_idx)} "
                  f"unknown-coast rows (buoys={buoys})")
            parts.append(self._fallback(X.loc[other_idx]))
        if not parts:
            return pd.DataFrame(
                {"pred_hs_m": [], "pred_tp_s": [], "pred_dp_deg": []},
                index=X.index,
            )
        out = pd.concat(parts).reindex(X.index)
        return out

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("lgbm_per_coast")
        self.east.save(path / "east")
        self.west.save(path / "west")
        (path / "router.json").write_text(json.dumps({
            "east_buoys": sorted(EAST_BUOYS),
            "west_buoys": sorted(WEST_BUOYS),
        }, indent=2))

    @classmethod
    def load(cls, path: Path):
        m = cls()
        m.east = LGBMEastCoast.load(path / "east")
        m.west = LGBMWestCoast.load(path / "west")
        return m
