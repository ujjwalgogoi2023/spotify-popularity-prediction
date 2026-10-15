"""SHAP summary for the audio-only model (isolated: TreeExplainer is the slow step)."""
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap
sys.path.insert(0, "src")
from data_prep import build, split, feature_columns
from models import RANDOM_STATE
from sklearn.ensemble import RandomForestRegressor

df = build(); tr, va, te = split(df)
cols = feature_columns(include_genre=False)
# Shallower forest keeps TreeExplainer tractable without changing the ranking.
rf = RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=20,
                           n_jobs=-1, random_state=RANDOM_STATE).fit(tr[cols], tr.popularity)
sample = va[cols].sample(600, random_state=RANDOM_STATE)
sv = shap.TreeExplainer(rf).shap_values(sample)
shap.summary_plot(sv, sample, show=False, plot_size=(8, 5))
plt.title("SHAP — audio features only", fontsize=11)
plt.tight_layout(); plt.savefig("figures/08_shap_summary.png", bbox_inches="tight"); plt.close()

mean_abs = pd.Series(abs(sv).mean(0), index=cols).sort_values(ascending=False)
print("=== SHAP mean |value| (audio only) ===")
print(mean_abs.round(4).to_string())
mean_abs.to_csv("data/processed/shap_importance.csv")
print("\nsaved figures/08_shap_summary.png")
