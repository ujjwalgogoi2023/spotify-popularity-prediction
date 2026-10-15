"""How much do the headline validation scores move with the random seed?

The selected model (tuned Random Forest) is refitted on five different train/validation
partitions, with the forest's own seed changed alongside, for both feature sets.

The test partition is held fixed at the canonical seed-42 split and never scored here:
only the 85% pool that train and validation share is re-partitioned, at the same sizes
and with the same popularity strata. Re-running `split` with a new seed would instead move
tracks from the already-scored test set into validation. Seed 42 uses the canonical
train/validation split, so its row reproduces `results_regression_trackA.csv`.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, "src")
from data_prep import PROCESSED, RANDOM_STATE, build, feature_columns, split
from models import evaluate, load_best_params, regression_models

SEEDS = [RANDOM_STATE, 1, 2, 3, 4]
BINS = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 101]


def resplit(pool: pd.DataFrame, n_val: int, seed: int):
    strata = pd.cut(pool.popularity, bins=BINS, labels=False)
    tr, va = train_test_split(pool, test_size=n_val, random_state=seed, stratify=strata)
    assert not set(tr.track_id) & set(va.track_id), "track_id leak: train/val"
    return tr, va


df = build()
tr42, va42, te42 = split(df)
pool = pd.concat([tr42, va42], ignore_index=True)

rows = []
for seed in SEEDS:
    tr, va = (tr42, va42) if seed == RANDOM_STATE else resplit(pool, len(va42), seed)
    assert not set(va.track_id) & set(te42.track_id), "validation overlaps test"
    for label, include_genre in [("audio only", False), ("audio+genre", True)]:
        cols = feature_columns(include_genre=include_genre)
        rf = regression_models(cols, load_best_params("regression", label))["Random Forest"]
        rf.set_params(m__random_state=seed)
        rf.fit(tr[cols], tr.popularity)
        m = evaluate(va.popularity, rf.predict(va[cols]))
        rows.append({"seed": seed, "features": label, "model": "Random Forest",
                     "n_train": len(tr), "n_val": len(va), **m})
        print(f"seed {seed:>2}  {label:<12} R2={m['R2']:.4f}  RMSE={m['RMSE']:.3f}", flush=True)

res = pd.DataFrame(rows)
res[["R2", "RMSE", "MAE"]] = res[["R2", "RMSE", "MAE"]].round(4)
res.to_csv(PROCESSED / "results_seed_variance.csv", index=False)

summary = res.groupby("features").R2.agg(["mean", "std", "min", "max"])
print(f"\nvalidation R2 across {len(SEEDS)} seeds (sample SD)\n{summary.round(4).to_string()}")
print(f"\nwritten: {PROCESSED / 'results_seed_variance.csv'}")
