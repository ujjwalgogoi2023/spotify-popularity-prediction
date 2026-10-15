"""
Model definitions and evaluation for the Spotify popularity study.

Every estimator is wrapped in a Pipeline whose preprocessing is fitted inside
cross-validation folds, so no validation or test statistic can reach a fitted
transformer. This is the structural defence against leakage; the track_id
assertions in data_prep.py are the other half.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BEST_PARAMS = Path(__file__).resolve().parent.parent / "data" / "processed" / "best_params.json"


def load_best_params(task: str, feature_set: str) -> dict[str, dict]:
    """Tuned hyperparameters from tuning.py, as {model name: {param: value}}.

    `task` is "regression" or "classification"; `feature_set` is "audio only" or
    "audio+genre". Returns {} when tuning has not been run, so every script still works
    with the hand-set defaults.
    """
    if not BEST_PARAMS.exists():
        return {}
    return json.loads(BEST_PARAMS.read_text()).get(task, {}).get(feature_set, {})


def _apply(models: dict, params: dict[str, dict] | None) -> dict:
    for name, p in (params or {}).items():
        if name in models:
            models[name].set_params(**{f"m__{k}": v for k, v in p.items()})
    return models


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Scale numerics, one-hot the genre column if it is present.

    Fitted per fold via the enclosing Pipeline, never on the full dataset.
    """
    categorical = [c for c in feature_cols if c == "track_genre"]
    numeric = [c for c in feature_cols if c != "track_genre"]

    transformers = [("num", StandardScaler(), numeric)]
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    return ColumnTransformer(transformers)


def regression_models(feature_cols: list[str],
                      params: dict[str, dict] | None = None) -> dict[str, Pipeline]:
    """The regression family, ordered from trivial baseline to best expected.

    `params` overrides the defaults per model, typically from `load_best_params`.
    """
    pre = lambda: make_preprocessor(feature_cols)

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline([("pre", pre()), ("m", LinearRegression())]),
        "Ridge": Pipeline([("pre", pre()), ("m", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
        "Lasso": Pipeline([("pre", pre()), ("m", Lasso(alpha=0.01, random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("pre", pre()), ("m", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("pre", pre()), ("m", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, random_state=RANDOM_STATE))]),
    }
    return _apply(models, params)


def classification_models(feature_cols: list[str],
                          params: dict[str, dict] | None = None) -> dict[str, Pipeline]:
    """The same three families as the regression set, for the binary framing (RQ4)."""
    pre = lambda: make_preprocessor(feature_cols)

    models = {
        "Logistic Regression": Pipeline([("pre", pre()), ("m", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("pre", pre()), ("m", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("pre", pre()), ("m", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=RANDOM_STATE))]),
    }
    return _apply(models, params)


class HurdleRegressor:
    """Two-stage model for the zero-inflated target.

    Stage 1 classifies whether a track has any popularity at all; stage 2 predicts
    how much, trained only on tracks that do. The prediction is the product of the
    two, which is the expected value under the hurdle formulation.

    This exists because 10.5% of tracks sit at exactly zero and those zeros are
    clustered by genre rather than scattered -- they look like availability gaps at
    scrape time, a different generating process from the popularity score itself.
    A single regressor is forced to average across both processes.
    """

    def __init__(self, feature_cols: list[str], random_state: int = RANDOM_STATE,
                 gb_params: dict | None = None):
        self.feature_cols = feature_cols
        self.clf = Pipeline([
            ("pre", make_preprocessor(feature_cols)),
            ("m", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, random_state=random_state)),
        ])
        self.reg = Pipeline([
            ("pre", make_preprocessor(feature_cols)),
            ("m", HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.08, random_state=random_state)),
        ])
        # Both stages are HistGB on the same features, so the capacity settings tuned
        # for the single-stage regressor transfer to each.
        if gb_params:
            for stage in (self.clf, self.reg):
                stage.set_params(**{f"m__{k}": v for k, v in gb_params.items()})

    def fit(self, X: pd.DataFrame, y: pd.Series):
        nonzero = (y > 0).astype(int)
        if nonzero.nunique() < 2:
            # No zero mass to model. Stage 1 would train on a single class and emit
            # degenerate probabilities, so the hurdle formulation does not apply --
            # this is the case for the zero-excluded sensitivity analysis.
            raise ValueError(
                "HurdleRegressor requires both zero and non-zero targets; "
                "the target contains only non-zero values."
            )
        self.clf.fit(X, nonzero)
        self.reg.fit(X.loc[y > 0], y.loc[y > 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_nonzero = self.clf.predict_proba(X)[:, 1]
        conditional = self.reg.predict(X)
        return p_nonzero * conditional


def evaluate(y_true, y_pred, n_features: int | None = None) -> dict:
    """Regression metrics, including adjusted R^2 when the feature count is known."""
    r2 = r2_score(y_true, y_pred)
    out = {
        "R2": r2,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
    if n_features is not None:
        n = len(y_true)
        out["Adj_R2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return out


def shuffle_check(model, X_train, y_train, X_val, y_val, random_state=RANDOM_STATE) -> float:
    """Refit on a shuffled target. R^2 must collapse to ~0; anything else means leakage."""
    rng = np.random.default_rng(random_state)
    y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
    model.fit(X_train, y_shuffled)
    return r2_score(y_val, model.predict(X_val))
