"""Final held-out test evaluation. The test set is touched exactly once, here.

Which model is evaluated is not decided here: the winner per feature set is read from
the validation results written by run_regression.py and run_classification.py, then
refitted on train+val with the tuned settings. The mean baseline and linear regression
are always reported alongside it as reference points.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "src")
from data_prep import build, split, feature_columns
from models import (regression_models, classification_models, HurdleRegressor, evaluate,
                    load_best_params, RANDOM_STATE)
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 140
OUT = "data/processed"
FEATURE_SETS = [("audio only", False), ("audio+genre", True)]

df = build(); tr, va, te = split(df)
# Validation was used for model selection; refit the chosen models on train+val.
trval = pd.concat([tr, va], ignore_index=True)

val_reg = pd.read_csv(f"{OUT}/results_regression_trackA.csv")
val_clf = pd.read_csv(f"{OUT}/results_classification.csv")
val_clf = val_clf[val_clf.threshold == "p75 of non-zero (primary)"]


def regressor(name, cols, params):
    if name == "Hurdle (2-stage)":
        return HurdleRegressor(cols, gb_params=params.get("Gradient Boosting"))
    return regression_models(cols, params)[name]


print("="*78); print("FINAL TEST-SET RESULTS (regression)"); print("="*78)
rows, winners = [], {}
for label, inc in FEATURE_SETS:
    cols = feature_columns(include_genre=inc)
    params = load_best_params("regression", label)
    cand = val_reg[(val_reg.features == label) & (val_reg.model != "Mean baseline")]
    best = cand.sort_values("R2", ascending=False).iloc[0].model
    print(f"{label}: validation winner = {best}")
    for name in dict.fromkeys(["Mean baseline", "Linear Regression", best]):
        est = regressor(name, cols, params).fit(trval[cols], trval.popularity)
        m = evaluate(te.popularity, est.predict(te[cols]), n_features=len(cols))
        rows.append({"features": label, "model": name, "selected": name == best, **m})
        if name == best:
            winners[label] = (name, est)
reg = pd.DataFrame(rows).round(4)
print(reg.to_string(index=False))
reg.to_csv(f"{OUT}/results_test_regression.csv", index=False)

print("\n" + "="*78); print("FINAL TEST-SET RESULTS (classification, popularity >= p75 of non-zero)"); print("="*78)
thr = trval.loc[trval.popularity > 0, "popularity"].quantile(0.75)
ytrval = (trval.popularity >= thr).astype(int); yte = (te.popularity >= thr).astype(int)
crows, cms = [], {}
for label, inc in FEATURE_SETS:
    cols = feature_columns(include_genre=inc)
    best = val_clf[val_clf.features == label].sort_values("PR_AUC", ascending=False).iloc[0].model
    clf = classification_models(cols, load_best_params("classification", label))[best]
    clf.fit(trval[cols], ytrval)
    p = clf.predict_proba(te[cols])[:, 1]; pred = (p >= .5).astype(int)
    crows.append({"features": label, "model": best,
        "ROC_AUC": roc_auc_score(yte, p), "PR_AUC": average_precision_score(yte, p),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0),
        "F1": f1_score(yte, pred, zero_division=0), "PR_baseline": yte.mean()})
    cms[label] = (best, confusion_matrix(yte, pred))
clf_res = pd.DataFrame(crows).round(4)
print(f"threshold = {thr:.0f}, positive rate = {yte.mean():.1%}")
print(clf_res.to_string(index=False))
clf_res.to_csv(f"{OUT}/results_test_classification.csv", index=False)

# --- Test confusion matrices ---------------------------------------------------
pd.DataFrame([{"features": label, "model": name, "TN": cm[0, 0], "FP": cm[0, 1],
               "FN": cm[1, 0], "TP": cm[1, 1]} for label, (name, cm) in cms.items()]
             ).to_csv(f"{OUT}/results_test_confusion.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (label, (name, cm)) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["not popular", "popular"], yticklabels=["not popular", "popular"])
    ax.set(title=f"{label} — {name}", xlabel="predicted", ylabel="actual")
fig.suptitle(f"Test-set confusion matrices (popularity ≥ {thr:.0f}, threshold 0.5)", fontsize=11)
plt.tight_layout(); plt.savefig("figures/09_confusion_matrix.png", bbox_inches="tight"); plt.close()
for label, (name, cm) in cms.items():
    print(f"\n{label} ({name}):  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

print("\n" + "="*78); print("PERMUTATION IMPORTANCE (test set, selected regressor)"); print("="*78)
sub = te.sample(3000, random_state=RANDOM_STATE)
for label, inc in FEATURE_SETS:
    cols = feature_columns(include_genre=inc)
    name, mdl = winners[label]
    # n_jobs=1: a full-depth forest is several GB, and parallel workers would each get a
    # memory-mapped copy on disk.
    pi = permutation_importance(mdl, sub[cols], sub.popularity, n_repeats=3, scoring="r2",
                                random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    print(f"\n--- {label} ({name}) ---"); print(imp.head(8).round(4).to_string())
    imp.to_csv(f"{OUT}/perm_importance_{'genre' if inc else 'audio'}.csv")
    # Figures are drawn from these same scores, so they always match the reported tables.
    std = pd.Series(pi.importances_std, index=cols)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp[::-1].plot.barh(ax=ax, color="#4C72B0", xerr=std.loc[imp.index[::-1]])
    ax.set(title=f"Permutation importance (test set) — {label}, {name}",
           xlabel="drop in R² when shuffled")
    fname = "07_permutation_genre.png" if inc else "06_permutation_audio.png"
    plt.tight_layout(); plt.savefig(f"figures/{fname}", bbox_inches="tight"); plt.close()
