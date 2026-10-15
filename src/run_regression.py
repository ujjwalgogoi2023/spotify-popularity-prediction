"""Fit every regression model on audio-only and audio+genre feature sets."""
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
sys.path.insert(0, "src")
from data_prep import build, split, feature_columns
from models import (regression_models, HurdleRegressor, evaluate, shuffle_check,
                    load_best_params)

def run(exclude_zeros=False, tag="Track A (all tracks)"):
    df = build(exclude_zeros=exclude_zeros)
    tr, va, te = split(df)
    print(f"\n{'='*78}\n{tag}  |  n={len(df):,}  ({len(tr):,}/{len(va):,}/{len(te):,})\n{'='*78}")
    rows = []
    for label, include_genre in [("audio only", False), ("audio+genre", True)]:
        cols = feature_columns(include_genre=include_genre)
        Xtr, ytr = tr[cols], tr.popularity
        Xva, yva = va[cols], va.popularity
        # Tuned on Track A's training partition; Track B reuses the same settings so the
        # two tracks differ only in the data, not in the models.
        params = load_best_params("regression", label)
        for name, model in regression_models(cols, params).items():
            model.fit(Xtr, ytr)
            m = evaluate(yva, model.predict(Xva))
            rows.append({"features": label, "model": name, **m})
        # Undefined when the zero mass has been removed (Track B).
        try:
            h = HurdleRegressor(cols, gb_params=params.get("Gradient Boosting")).fit(Xtr, ytr)
            rows.append({"features": label, "model": "Hurdle (2-stage)",
                         **evaluate(yva, h.predict(Xva))})
        except ValueError as e:
            print(f"  [skipped] Hurdle ({label}): {e}")
    res = pd.DataFrame(rows)
    res["R2"] = res.R2.round(4); res["RMSE"] = res.RMSE.round(3); res["MAE"] = res.MAE.round(3)
    print(res.to_string(index=False))
    return df, tr, va, te, res

df, tr, va, te, resA = run(False, "TRACK A - all tracks (zero mass retained)")
resA.to_csv("data/processed/results_regression_trackA.csv", index=False)

_, _, _, _, resB = run(True, "TRACK B - zero-popularity tracks excluded (sensitivity)")
resB.to_csv("data/processed/results_regression_trackB.csv", index=False)

# Leakage sanity check on the strongest configuration.
print(f"\n{'='*78}\nSHUFFLE CHECK (target permuted -> R2 must be ~0)\n{'='*78}")
cols = feature_columns(include_genre=True)
gb = regression_models(cols, load_best_params("regression", "audio+genre"))["Gradient Boosting"]
r2s = shuffle_check(gb, tr[cols], tr.popularity, va[cols], va.popularity)
print(f"shuffled-target R2 = {r2s:.5f}   -> {'PASS' if abs(r2s) < 0.02 else 'FAIL - LEAKAGE'}")
