"""External validation and the era confound (DA1 sections 4.3 and 8.5).

Part A -- transfer. The audio-only classifier is trained on the primary dataset
(train+val; the primary test set is not touched) and scored on the Hit Predictor
tracks against their independent hit/flop label. The targets differ -- Spotify
popularity at scrape time versus a Billboard-based hit label -- so this tests whether
the direction of the audio signal generalises, not whether the scores match.

Part B -- era. The primary dataset has no release year, so whether audio features
partly proxy for release date could only be acknowledged there. The Hit Predictor data
carries a decade for every track, which allows four direct measurements:
  1. how far the audio features drift across decades;
  2. how well audio alone predicts a track's decade;
  3. whether telling the hit model the decade improves it (era-specific hit profiles);
  4. leave-one-decade-out versus random CV -- how much accuracy is lost when the model
     must predict an era it has not seen.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "src")
from data_prep import PROCESSED, build, split, feature_columns
from models import (classification_models, regression_models, load_best_params,
                    RANDOM_STATE)
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import (StratifiedKFold, KFold, LeaveOneGroupOut,
                                     cross_val_predict, cross_val_score)

sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 140
OUT = "data/processed"
COLS = feature_columns(include_genre=False, include_explicit=False)
rng = np.random.default_rng(RANDOM_STATE)


def bootstrap_auc(y, p, n=300):
    y, p = np.asarray(y), np.asarray(p)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    aucs = [roc_auc_score(y[i], p[i]) for i in idx if y[i].min() != y[i].max()]
    return np.percentile(aucs, [2.5, 97.5])


ext = pd.read_csv(PROCESSED / "external_clean.csv")
decades = sorted(ext.decade.unique())

# ================================================================== A. transfer
df = build(); tr, va, _ = split(df)
trval = pd.concat([tr, va], ignore_index=True)
thr = trval.loc[trval.popularity > 0, "popularity"].quantile(0.75)

val_clf = pd.read_csv(f"{OUT}/results_classification.csv")
val_clf = val_clf[(val_clf.threshold == "p75 of non-zero (primary)")
                  & (val_clf.features == "audio only")]
name = val_clf.sort_values("PR_AUC", ascending=False).iloc[0].model
model = classification_models(COLS, load_best_params("classification", "audio only"))[name]
model.fit(trval[COLS], (trval.popularity >= thr).astype(int))
ext["p_popular"] = model.predict_proba(ext[COLS])[:, 1]

print("=" * 78)
print(f"A. TRANSFER: {name} trained on primary (popularity >= {thr:.0f}), "
      f"scored on {len(ext):,} external tracks")
print("=" * 78)
rows = []
for scope, part in [("all decades", ext)] + [(str(d), ext[ext.decade == d]) for d in decades]:
    lo, hi = bootstrap_auc(part.target, part.p_popular)
    rows.append({"scope": scope, "model": name, "n": len(part),
                 "hit_rate": part.target.mean(),
                 "ROC_AUC": roc_auc_score(part.target, part.p_popular),
                 "AUC_CI_low": lo, "AUC_CI_high": hi,
                 "PR_AUC": average_precision_score(part.target, part.p_popular)})
transfer = pd.DataFrame(rows).round(4)
print(transfer.to_string(index=False))
transfer.to_csv(f"{OUT}/results_external.csv", index=False)

# ======================================================================= B. era
print("\n" + "=" * 78); print("B. ERA CONFOUND (within the external dataset)"); print("=" * 78)
era = []
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
clf_params = load_best_params("classification", "audio only")

# B1 -- drift: decade means in units of the pooled standard deviation.
DRIFT = ["loudness", "energy", "acousticness", "danceability", "valence", "log_duration"]
z = (ext.groupby("decade")[DRIFT].mean() - ext[DRIFT].mean()) / ext[DRIFT].std()
print("\nB1. Feature drift (decade mean, in pooled SDs):")
print(z.round(2).to_string())
for f in DRIFT:
    era.append({"test": "B1 drift 1960s->2010s (SD units)", "variant": f,
                "value": z.loc[decades[-1], f] - z.loc[decades[0], f]})
fig, ax = plt.subplots(figsize=(8, 4.5))
for f in DRIFT:
    ax.plot(z.index, z[f], marker="o", label=f)
ax.axhline(0, color="grey", lw=0.8)
ax.set(title="Audio feature drift by decade (Hit Predictor dataset)",
       xlabel="decade", ylabel="decade mean − overall mean (SDs)", xticks=decades)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig("figures/10_feature_drift_by_decade.png", bbox_inches="tight")
plt.close()

# B2 -- how well does audio alone recover the decade?
gb_reg = regression_models(COLS, load_best_params("regression", "audio only"))["Gradient Boosting"]
r2 = cross_val_score(gb_reg, ext[COLS], ext.decade, scoring="r2",
                     cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)).mean()
print(f"\nB2. Decade predicted from audio alone: 5-fold CV R2 = {r2:.3f}")
era.append({"test": "B2 decade from audio (5-fold CV R2)", "variant": "Gradient Boosting",
            "value": r2})

# B3 -- does knowing the decade improve the hit model?
print("\nB3. Hit/flop from audio, with and without decade (5-fold CV ROC-AUC):")
for mname in ["Logistic Regression", "Gradient Boosting"]:
    aucs = {}
    for variant, cols in [("audio", COLS), ("audio+decade", COLS + ["decade"])]:
        m = classification_models(cols, clf_params)[mname]
        aucs[variant] = cross_val_score(m, ext[cols], ext.target, cv=skf,
                                        scoring="roc_auc").mean()
        era.append({"test": "B3 hit AUC (5-fold CV)", "variant": f"{mname}, {variant}",
                    "value": aucs[variant]})
    gain = aucs["audio+decade"] - aucs["audio"]
    era.append({"test": "B3 AUC gain from decade", "variant": mname, "value": gain})
    print(f"  {mname:20s} audio={aucs['audio']:.4f}  +decade={aucs['audio+decade']:.4f}  "
          f"gain={gain:+.4f}")

# B4 -- leave-one-decade-out versus random CV, same model.
gb = classification_models(COLS, clf_params)["Gradient Boosting"]
p_rand = cross_val_predict(gb, ext[COLS], ext.target, cv=skf, method="predict_proba")[:, 1]
p_lodo = cross_val_predict(gb, ext[COLS], ext.target, groups=ext.decade,
                           cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
print("\nB4. Random 5-fold CV vs leave-one-decade-out (Gradient Boosting, ROC-AUC):")
lodo_rows = []
for d in decades:
    m = (ext.decade == d).to_numpy()
    a_r, a_l = roc_auc_score(ext.target[m], p_rand[m]), roc_auc_score(ext.target[m], p_lodo[m])
    lodo_rows.append({"decade": d, "random_cv": a_r, "leave_decade_out": a_l})
    era.append({"test": "B4 hit AUC by decade", "variant": f"{d} random CV", "value": a_r})
    era.append({"test": "B4 hit AUC by decade", "variant": f"{d} decade held out", "value": a_l})
lodo = pd.DataFrame(lodo_rows).set_index("decade")
print(lodo.round(4).to_string())
mean_gap = (lodo.random_cv - lodo.leave_decade_out).mean()
era.append({"test": "B4 mean AUC lost when decade unseen", "variant": "Gradient Boosting",
            "value": mean_gap})
print(f"  mean AUC lost when the decade is unseen: {mean_gap:+.4f}")

era = pd.DataFrame(era); era["value"] = era.value.round(4)
era.to_csv(f"{OUT}/results_era.csv", index=False)

# Figure 11 -- transfer and within-dataset AUC by decade.
per = transfer[transfer.scope != "all decades"].assign(decade=lambda d: d.scope.astype(int))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(per.decade, per.ROC_AUC, yerr=[per.ROC_AUC - per.AUC_CI_low,
            per.AUC_CI_high - per.ROC_AUC], marker="o", capsize=3,
            label=f"transfer: trained on primary ({name})")
ax.plot(lodo.index, lodo.random_cv, marker="s", label="within Hit Predictor: random 5-fold CV")
ax.plot(lodo.index, lodo.leave_decade_out, marker="^",
        label="within Hit Predictor: decade held out")
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set(title="Hit/flop ROC-AUC by decade", xlabel="decade", ylabel="ROC-AUC", xticks=decades)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/11_transfer_auc_by_decade.png", bbox_inches="tight")
plt.close()

print(f"\nwritten: {OUT}/results_external.csv, {OUT}/results_era.csv")
print("written: figures/10_feature_drift_by_decade.png, figures/11_transfer_auc_by_decade.png")
