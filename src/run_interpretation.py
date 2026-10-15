"""EDA + interpretation figures for the report."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "src")
from data_prep import load_raw, build, split, feature_columns, AUDIO_FEATURES
from models import make_preprocessor, RANDOM_STATE
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 140
FIG = "figures"

raw = load_raw()
df  = build(); tr, va, te = split(df)

# --- F1: duplicate structure -------------------------------------------------
vc = raw.track_id.value_counts()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
vc.value_counts().sort_index().plot.bar(ax=ax[0], color="#4C72B0")
ax[0].set(title="Genre assignments per track (raw file)", xlabel="# genres", ylabel="tracks")
ax[0].set_yscale("log")
ax[1].bar(["raw rows","unique tracks"], [len(raw), raw.track_id.nunique()],
          color=["#C44E52","#55A868"])
ax[1].set(title=f"Duplicate inflation: {100*(len(raw)-raw.track_id.nunique())/len(raw):.1f}% of rows")
for i,v in enumerate([len(raw), raw.track_id.nunique()]): ax[1].text(i, v, f"{v:,}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(f"{FIG}/01_duplicates.png", bbox_inches="tight"); plt.close()

# --- F2: target distribution -------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(raw.popularity, bins=101, color="#4C72B0"); ax[0].set(title="Popularity (raw, with duplicates)", xlabel="popularity")
ax[1].hist(df.popularity, bins=101, color="#55A868"); ax[1].set(title=f"Popularity (deduplicated) — {100*df.is_zero_pop.mean():.1f}% at zero", xlabel="popularity")
plt.tight_layout(); plt.savefig(f"{FIG}/02_target_distribution.png", bbox_inches="tight"); plt.close()

# --- F3: zeros are genre-clustered, not random -------------------------------
zg = (raw.assign(z=raw.popularity==0).groupby("track_genre").z.mean()*100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
pd.concat([zg.head(12), zg.tail(12)]).plot.barh(ax=ax, color=["#C44E52"]*12+["#55A868"]*12)
ax.set(title="Zero-popularity rate by genre (top/bottom 12)", xlabel="% of tracks with popularity = 0")
ax.invert_yaxis(); plt.tight_layout(); plt.savefig(f"{FIG}/03_zeros_by_genre.png", bbox_inches="tight"); plt.close()

# --- F4: correlation heatmap -------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[AUDIO_FEATURES+["popularity"]].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, annot_kws={"size":7}, cbar_kws={"shrink":.8})
ax.set_title("Audio feature correlations (deduplicated)")
plt.tight_layout(); plt.savefig(f"{FIG}/04_correlation.png", bbox_inches="tight"); plt.close()

# --- F5: genre mean popularity ----------------------------------------------
gp = df.groupby("track_genre").popularity.mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
pd.concat([gp.head(12), gp.tail(12)]).plot.barh(ax=ax, color=["#55A868"]*12+["#C44E52"]*12)
ax.set(title="Mean popularity by genre (top/bottom 12)", xlabel="mean popularity")
ax.invert_yaxis(); plt.tight_layout(); plt.savefig(f"{FIG}/05_genre_popularity.png", bbox_inches="tight"); plt.close()

# --- VIF ---------------------------------------------------------------------
vif_cols = ["danceability","energy","loudness","speechiness","acousticness",
            "instrumentalness","liveness","valence","tempo"]
Xv = df[vif_cols].dropna()
vif = pd.DataFrame({"feature": vif_cols,
    "VIF": [variance_inflation_factor(Xv.values, i) for i in range(len(vif_cols))]})
vif = vif.sort_values("VIF", ascending=False)
print("=== VIF (multicollinearity) ===\n" + vif.round(2).to_string(index=False))
vif.to_csv("data/processed/vif.csv", index=False)

# --- Permutation importance (audio only + audio+genre) -----------------------
for label, inc, fname in [("audio only", False, "06_permutation_audio.png"),
                          ("audio+genre", True, "07_permutation_genre.png")]:
    cols = feature_columns(include_genre=inc)
    mdl = Pipeline([("pre", make_preprocessor(cols)),
                    ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                         n_jobs=-1, random_state=RANDOM_STATE))]).fit(tr[cols], tr.popularity)
    sub = va.sample(4000, random_state=RANDOM_STATE)
    pi = permutation_importance(mdl, sub[cols], sub.popularity, n_repeats=5,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="#4C72B0", xerr=pd.Series(pi.importances_std, index=cols).loc[imp.index])
    ax.set(title=f"Permutation importance — {label}", xlabel="drop in R² when shuffled")
    plt.tight_layout(); plt.savefig(f"{FIG}/{fname}", bbox_inches="tight"); plt.close()
    print(f"\n=== Permutation importance ({label}) — top 6 ===")
    print(imp.sort_values(ascending=False).head(6).round(4).to_string())

# --- SHAP on the audio-only model (interpretable without 114 genre dummies) ---
import shap
cols = feature_columns(include_genre=False)
rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=-1,
                           random_state=RANDOM_STATE).fit(tr[cols], tr.popularity)
sample = va[cols].sample(1500, random_state=RANDOM_STATE)
sv = shap.TreeExplainer(rf).shap_values(sample)
shap.summary_plot(sv, sample, show=False, plot_size=(8,5))
plt.title("SHAP — audio features only", fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/08_shap_summary.png", bbox_inches="tight"); plt.close()
print("\nfigures written to figures/")
