"""Hyperparameter search, cross-validated inside train.csv only.

Validation and test partitions are never loaded here, so the search cannot select on
them. Each candidate is the full Pipeline from models.py, so the scaler and encoder are
refitted inside every fold.

The search is deliberately light (a few minutes per model): the study's question is
the ceiling of audio features, not leaderboard accuracy, so tuning exists to show the
reported numbers are not an artefact of arbitrary settings. Forests are searched with
60 trees and refitted with 300 -- tree count changes the score little and the ranking
of the other settings even less.
"""
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from data_prep import PROCESSED, feature_columns
from models import BEST_PARAMS, RANDOM_STATE, classification_models, regression_models
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold)

SEARCH_TREES = 60

REGRESSION_SPACE = {
    "Ridge": ("grid", {"alpha": list(np.logspace(-2, 3, 11))}),
    "Lasso": ("grid", {"alpha": list(np.logspace(-3, 0, 7))}),
    "Random Forest": ("random", {
        "max_features": [0.33, 0.5, 0.7, 1.0],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_depth": [None, 20, 30]}),
    "Gradient Boosting": ("random", {
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_iter": [300, 600]}),
}
CLASSIFICATION_SPACE = {
    "Logistic Regression": ("grid", {"C": list(np.logspace(-3, 2, 6))}),
    "Random Forest": REGRESSION_SPACE["Random Forest"],
    "Gradient Boosting": REGRESSION_SPACE["Gradient Boosting"],
}
N_ITER = {"Random Forest": 8, "Gradient Boosting": 12}


def _plain(v):
    """numpy scalars -> JSON-serialisable Python values."""
    return v.item() if isinstance(v, np.generic) else v


def search(model, kind, space, name, X, y, cv, scoring):
    grid = {f"m__{k}": v for k, v in space.items()}
    if name == "Random Forest":
        model.set_params(m__n_estimators=SEARCH_TREES)
    if kind == "grid":
        s = GridSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=1)
    else:
        s = RandomizedSearchCV(model, grid, n_iter=N_ITER[name], cv=cv, scoring=scoring,
                               n_jobs=1, random_state=RANDOM_STATE)
    s.fit(X, y)
    best = {k.removeprefix("m__"): _plain(v) for k, v in s.best_params_.items()}
    return best, s.best_score_, s.cv_results_["mean_test_score"]


def main():
    tr = pd.read_csv(PROCESSED / "train.csv")
    # Same definition as run_classification.py: 75th percentile of non-zero popularity.
    thr = tr.loc[tr.popularity > 0, "popularity"].quantile(0.75)
    y_reg, y_clf = tr.popularity, (tr.popularity >= thr).astype(int)

    best_params = {"regression": {}, "classification": {}}
    rows = []
    jobs = [
        ("regression", REGRESSION_SPACE, regression_models, y_reg, "r2",
         KFold(3, shuffle=True, random_state=RANDOM_STATE)),
        ("classification", CLASSIFICATION_SPACE, classification_models, y_clf,
         "average_precision", StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)),
    ]
    for task, spaces, factory, y, scoring, cv in jobs:
        for label, inc in [("audio only", False), ("audio+genre", True)]:
            cols = feature_columns(include_genre=inc)
            models = factory(cols)
            best_params[task][label] = {}
            for name, (kind, space) in spaces.items():
                t = time.time()
                best, score, all_scores = search(models[name], kind, space, name,
                                                 tr[cols], y, cv, scoring)
                best_params[task][label][name] = best
                rows.append({"task": task, "features": label, "model": name,
                             "scoring": scoring, "best_cv": round(score, 4),
                             "worst_cv": round(float(np.nanmin(all_scores)), 4),
                             "candidates": len(all_scores),
                             "seconds": round(time.time() - t), "best_params": json.dumps(best)})
                print(f"[{task:14s}] {label:11s} {name:20s} {scoring}={score:.4f}  "
                      f"({len(all_scores)} candidates, {time.time()-t:.0f}s)  {best}",
                      flush=True)

    BEST_PARAMS.write_text(json.dumps(best_params, indent=2))
    pd.DataFrame(rows).to_csv(PROCESSED / "results_tuning.csv", index=False)
    print(f"\nwritten: {BEST_PARAMS}\nwritten: {PROCESSED / 'results_tuning.csv'}")


if __name__ == "__main__":
    main()
