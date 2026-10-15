"""Binary 'popular vs not' framing, for comparison against the regression models (RQ4)."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from data_prep import build, split, feature_columns
from models import classification_models, load_best_params
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

df = build(exclude_zeros=False)
tr, va, te = split(df)

# Threshold is a modelling choice, not a property of the data. The primary cut is the
# 75th percentile of NON-ZERO popularity; a second cut is reported as a sensitivity check.
nz = tr.loc[tr.popularity > 0, "popularity"]
THRESHOLDS = {"p75 of non-zero (primary)": nz.quantile(0.75),
              "p90 of non-zero (sensitivity)": nz.quantile(0.90)}

rows = []
for tname, thr in THRESHOLDS.items():
    ytr = (tr.popularity >= thr).astype(int)
    yva = (va.popularity >= thr).astype(int)
    print(f"\n{'='*78}\n{tname}: popularity >= {thr:.0f}   "
          f"positive rate = {ytr.mean():.1%}\n{'='*78}")
    for label, inc in [("audio only", False), ("audio+genre", True)]:
        cols = feature_columns(include_genre=inc)
        Xtr, Xva = tr[cols], va[cols]
        # Tuned at the primary threshold; the p90 check reuses those settings.
        models = classification_models(cols, load_best_params("classification", label))
        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            proba = mdl.predict_proba(Xva)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({"threshold": tname, "features": label, "model": name,
                "ROC_AUC": roc_auc_score(yva, proba),
                "PR_AUC": average_precision_score(yva, proba),
                "Precision": precision_score(yva, pred, zero_division=0),
                "Recall": recall_score(yva, pred, zero_division=0),
                "F1": f1_score(yva, pred, zero_division=0)})

res = pd.DataFrame(rows)
for c in ["ROC_AUC","PR_AUC","Precision","Recall","F1"]:
    res[c] = res[c].round(4)
print(res.to_string(index=False))
res.to_csv("data/processed/results_classification.csv", index=False)

# Validation confusion matrix for the best configuration at the primary threshold.
# The held-out test equivalent is produced once, in run_final_test.py.
primary = res[(res.threshold == "p75 of non-zero (primary)") & (res.features == "audio+genre")]
best_name = primary.sort_values("PR_AUC", ascending=False).iloc[0].model
thr = THRESHOLDS["p75 of non-zero (primary)"]
cols = feature_columns(include_genre=True)
best = classification_models(cols, load_best_params("classification", "audio+genre"))[best_name]
best.fit(tr[cols], (tr.popularity>=thr).astype(int))
yva = (va.popularity>=thr).astype(int)
cm = confusion_matrix(yva, (best.predict_proba(va[cols])[:,1]>=0.5).astype(int))
print(f"\nValidation confusion matrix ({best_name}, audio+genre, p75):\n"
      f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}\n  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
