"""Candidate CSC v1 models — common fit/predict/save/load API.

Every model exposes the same interface:

    m = SomeModel()
    m.fit(X_train_df, y_train_series_or_df)
    preds = m.predict(X_test_df)        # DataFrame with target columns
    m.save(path: Path)                   # writes a directory of artifacts
    m = SomeModel.load(path)             # class method

Targets are handled as a dict of three independent regressions:
  obs_hs_m    → one regressor
  obs_tp_s    → one regressor
  (obs_sin_dp, obs_cos_dp) → two regressors; direction reconstructed via atan2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from csc.features import categorical_columns, feature_columns


TARGET_HS = "obs_hs_m"
TARGET_TP = "obs_tp_s"
TARGET_SIN = "obs_sin_dp"
TARGET_COS = "obs_cos_dp"


def split_targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "hs": df[TARGET_HS],
        "tp": df[TARGET_TP],
        "sin_dp": df[TARGET_SIN],
        "cos_dp": df[TARGET_COS],
    }


def reconstruct_dp(sin_pred: np.ndarray, cos_pred: np.ndarray) -> np.ndarray:
    """atan2 reconstruction; returns degrees in [0, 360)."""
    ang = np.rad2deg(np.arctan2(sin_pred, cos_pred))
    return (ang + 360.0) % 360.0


# ─── Baselines ────────────────────────────────────────────────────────────

class RawGFSBaseline:
    """Always predicts the raw GFS analysis (if available)."""
    name = "raw_gfs"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "pred_hs_m": X["gfs_wave_height"],
            "pred_tp_s": X["gfs_wave_period"],
            "pred_dp_deg": X["gfs_wave_direction"],
        }, index=X.index)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("raw_gfs")

    @classmethod
    def load(cls, path: Path):
        return cls()


class RawEUROBaseline:
    name = "raw_euro"
    def fit(self, X, y): return self
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "pred_hs_m": X["euro_wave_height"],
            "pred_tp_s": X["euro_wave_period"],
            "pred_dp_deg": X["euro_wave_direction"],
        }, index=X.index)
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("raw_euro")
    @classmethod
    def load(cls, path: Path):
        return cls()


class RawGFSPrimaryBaseline:
    """v2 baseline — predicts GFS's primary-swell partition (`swell_wave_*`).

    When training against NDBC primary-swell Hs, the raw-GFS comparison
    must also use GFS's primary-swell output, otherwise the baseline is
    apples-to-oranges (combined sea state vs surfable partition). Falls
    back to the combined `gfs_wave_*` values when the partition column
    is missing or NaN (rare — Open-Meteo does expose GFS swell partitions
    for every buoy we track)."""
    name = "raw_gfs_primary"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        def _fallback(primary: str, combined: str) -> pd.Series:
            if primary in X.columns:
                return X[primary].fillna(X.get(combined))
            return X[combined]
        return pd.DataFrame({
            "pred_hs_m": _fallback("gfs_swell_wave_height", "gfs_wave_height"),
            "pred_tp_s": _fallback("gfs_swell_wave_period", "gfs_wave_period"),
            "pred_dp_deg": _fallback("gfs_swell_wave_direction",
                                     "gfs_wave_direction"),
        }, index=X.index)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("raw_gfs_primary")

    @classmethod
    def load(cls, path: Path):
        return cls()


class RawEUROPrimaryBaseline:
    """v2 baseline — EURO has NO swell partitions in Open-Meteo's ECMWF WAM
    feed, so for a primary-swell target there is no clean way to make EURO
    comparable. We predict the combined `euro_wave_*` values unchanged and
    note the limitation in the model name; the retrain report should flag
    that this baseline will overshoot the primary-swell target."""
    name = "raw_euro_primary"

    def fit(self, X, y):
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "pred_hs_m": X["euro_wave_height"],
            "pred_tp_s": X["euro_wave_period"],
            "pred_dp_deg": X["euro_wave_direction"],
        }, index=X.index)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("raw_euro_primary")

    @classmethod
    def load(cls, path: Path):
        return cls()


class MeanBaseline:
    """Simple 50/50 mean of raw GFS and raw Euro."""
    name = "mean"
    def fit(self, X, y): return self
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Circular mean for direction
        sin_avg = 0.5 * (np.sin(np.deg2rad(X["gfs_wave_direction"])) +
                         np.sin(np.deg2rad(X["euro_wave_direction"])))
        cos_avg = 0.5 * (np.cos(np.deg2rad(X["gfs_wave_direction"])) +
                         np.cos(np.deg2rad(X["euro_wave_direction"])))
        dp = reconstruct_dp(sin_avg.values, cos_avg.values)
        return pd.DataFrame({
            "pred_hs_m": 0.5 * (X["gfs_wave_height"] + X["euro_wave_height"]),
            "pred_tp_s": 0.5 * (X["gfs_wave_period"] + X["euro_wave_period"]),
            "pred_dp_deg": dp,
        }, index=X.index)
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("mean")
    @classmethod
    def load(cls, path: Path):
        return cls()


class PersistenceBaseline:
    """For a given valid_utc, predict the most recent observed value
    available at t-6h. Requires an observation time series at fit time."""
    name = "persistence"
    def __init__(self):
        self.obs = None
    def fit(self, X, y, obs_df: pd.DataFrame | None = None):
        self.obs = obs_df
        return self
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.obs is None:
            return pd.DataFrame({
                "pred_hs_m": X["gfs_wave_height"],
                "pred_tp_s": X["gfs_wave_period"],
                "pred_dp_deg": X["gfs_wave_direction"],
            }, index=X.index)
        # Nearest-obs persistence, within 6h before valid_utc
        obs = self.obs.sort_values(["buoy_id", "valid_utc"])
        out_rows = []
        for _, row in X.iterrows():
            mask = ((obs["buoy_id"] == row["buoy_id"]) &
                    (obs["valid_utc"] <= row["valid_utc"] - pd.Timedelta("6h")))
            prior = obs[mask].tail(1)
            if prior.empty:
                out_rows.append((np.nan, np.nan, np.nan))
            else:
                out_rows.append((prior["obs_hs_m"].iloc[0],
                                 prior["obs_tp_s"].iloc[0],
                                 prior["obs_dp_deg"].iloc[0]))
        a = np.array(out_rows, dtype=float)
        return pd.DataFrame({
            "pred_hs_m": a[:, 0], "pred_tp_s": a[:, 1], "pred_dp_deg": a[:, 2],
        }, index=X.index)
    def save(self, path: Path): path.mkdir(parents=True, exist_ok=True); (path/"kind").write_text("persistence")
    @classmethod
    def load(cls, path: Path): return cls()


# ─── Ridge MOS ────────────────────────────────────────────────────────────

@dataclass
class _MatrixSpec:
    feature_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)


class RidgeMOS:
    """Per-target Ridge regression on the full feature set. Categoricals
    are one-hot encoded. Missing numeric values are median-imputed."""
    name = "ridge_mos"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.spec = _MatrixSpec()
        self.scaler = None
        self.imputer = None
        self.encoder = None
        self.models: dict[str, Any] = {}

    def _prep(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        num = X[self.spec.feature_cols].astype(float).to_numpy()
        cat = X[self.spec.cat_cols].astype(str).to_numpy()
        if fit:
            self.imputer = SimpleImputer(strategy="median")
            num_imp = self.imputer.fit_transform(num)
            self.scaler = StandardScaler()
            num_std = self.scaler.fit_transform(num_imp)
            try:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            cat_enc = self.encoder.fit_transform(cat)
        else:
            num_imp = self.imputer.transform(num)
            num_std = self.scaler.transform(num_imp)
            cat_enc = self.encoder.transform(cat)
        return np.hstack([num_std, cat_enc])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        from sklearn.linear_model import Ridge
        self.spec = _MatrixSpec(
            feature_cols=feature_columns(use_partition_features=True),
            cat_cols=categorical_columns(),
        )
        Z = self._prep(X, fit=True)
        targets = split_targets(y)
        for tgt, series in targets.items():
            mask = ~series.isna()
            m = Ridge(alpha=self.alpha)
            m.fit(Z[mask], series[mask].to_numpy())
            self.models[tgt] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self._prep(X, fit=False)
        hs = self.models["hs"].predict(Z)
        tp = self.models["tp"].predict(Z)
        sp = self.models["sin_dp"].predict(Z)
        cp = self.models["cos_dp"].predict(Z)
        dp = reconstruct_dp(sp, cp)
        return pd.DataFrame({
            "pred_hs_m": hs, "pred_tp_s": tp, "pred_dp_deg": dp,
        }, index=X.index)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("ridge_mos")
        joblib.dump({
            "alpha": self.alpha,
            "spec": self.spec,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "encoder": self.encoder,
            "models": self.models,
        }, path / "model.joblib")

    @classmethod
    def load(cls, path: Path):
        obj = joblib.load(path / "model.joblib")
        m = cls(alpha=obj["alpha"])
        m.spec = obj["spec"]
        m.scaler = obj["scaler"]
        m.imputer = obj["imputer"]
        m.encoder = obj["encoder"]
        m.models = obj["models"]
        return m


# ─── LightGBM (global, categorical-aware) ─────────────────────────────────

class LGBMCSC:
    name = "lgbm"

    def __init__(self, n_estimators: int = 600, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_data_in_leaf: int = 100):
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf,
            feature_fraction=0.9, verbose=-1,
        )
        self.feature_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.models: dict[str, Any] = {}

    def _Xy(self, X: pd.DataFrame) -> pd.DataFrame:
        use = X[self.feature_cols + self.cat_cols].copy()
        for c in self.cat_cols:
            use[c] = use[c].astype("category")
        return use

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        import lightgbm as lgb
        self.feature_cols = feature_columns(use_partition_features=True)
        self.cat_cols = categorical_columns()
        Xp = self._Xy(X)
        targets = split_targets(y)
        for tgt, series in targets.items():
            mask = ~series.isna()
            m = lgb.LGBMRegressor(**self.params)
            m.fit(Xp[mask], series[mask].to_numpy(),
                  categorical_feature=self.cat_cols)
            self.models[tgt] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = self._Xy(X)
        hs = self.models["hs"].predict(Xp)
        tp = self.models["tp"].predict(Xp)
        sp = self.models["sin_dp"].predict(Xp)
        cp = self.models["cos_dp"].predict(Xp)
        dp = reconstruct_dp(sp, cp)
        return pd.DataFrame({
            "pred_hs_m": hs, "pred_tp_s": tp, "pred_dp_deg": dp,
        }, index=X.index)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "kind").write_text("lgbm")
        joblib.dump({
            "params": self.params,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "models": self.models,
        }, path / "model.joblib")

    @classmethod
    def load(cls, path: Path):
        obj = joblib.load(path / "model.joblib")
        m = cls()                         # default ctor; restore full params dict
        m.params = obj["params"]
        m.feature_cols = obj["feature_cols"]
        m.cat_cols = obj["cat_cols"]
        m.models = obj["models"]
        return m


# ─── Registry ─────────────────────────────────────────────────────────────

def all_candidates() -> dict[str, Any]:
    """Return the constructors for every candidate that appears in the
    bakeoff. Baselines are included alongside trained models so the
    experiment table has a row for each."""
    from csc.funplus import LGBMFunPlus
    from csc.per_coast import LGBMEastCoast, LGBMWestCoast, PerCoastRouter
    return {
        "raw_gfs": RawGFSBaseline,
        "raw_euro": RawEUROBaseline,
        "mean": MeanBaseline,
        "persistence": PersistenceBaseline,
        "ridge_mos": RidgeMOS,
        "lgbm": LGBMCSC,
        "funplus": LGBMFunPlus,
        "lgbm_east": LGBMEastCoast,
        "lgbm_west": LGBMWestCoast,
        "lgbm_per_coast": PerCoastRouter,
    }


def load_model(path: Path):
    """Generic loader — dispatches on the 'kind' file in the artifact dir."""
    from csc.funplus import LGBMFunPlus
    from csc.per_coast import LGBMEastCoast, LGBMWestCoast, PerCoastRouter
    kind = (path / "kind").read_text().strip()
    return {
        "raw_gfs": RawGFSBaseline,
        "raw_euro": RawEUROBaseline,
        "raw_gfs_primary": RawGFSPrimaryBaseline,
        "raw_euro_primary": RawEUROPrimaryBaseline,
        "mean": MeanBaseline,
        "persistence": PersistenceBaseline,
        "ridge_mos": RidgeMOS,
        "lgbm": LGBMCSC,
        "funplus": LGBMFunPlus,
        "lgbm_east": LGBMEastCoast,
        "lgbm_west": LGBMWestCoast,
        "lgbm_per_coast": PerCoastRouter,
    }[kind].load(path)
