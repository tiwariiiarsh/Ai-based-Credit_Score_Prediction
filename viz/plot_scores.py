import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scoring.impute import load_gmm, impute_features

gmm_bundle = load_gmm("models/gmm_model.pkl")
with open("models/xgb_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model     = bundle["model"]
feat_cols = bundle["feature_cols"]

# Load new users with missing data
df = pd.read_csv("data/new_users.csv")
print(f"Scoring {len(df)} new users...")
print(f"Missing values: {df.isnull().sum().sum()}")

scores = []
for i, row in df.iterrows():
    feat_dict = {}
    for col in feat_cols:
        if col in df.columns:
            val = row[col]
            feat_dict[col] = np.nan if pd.isna(val) else float(val)
        else:
            feat_dict[col] = np.nan

    imp   = impute_features(feat_dict, gmm_bundle)
    vec   = np.array([imp.get(c, 0) for c in feat_cols]).reshape(1, -1)
    score = int(np.clip(round(float(model.predict(vec)[0])), 300, 900))
    scores.append(score)
    if (i+1) % 100 == 0:
        print(f"  Scored {i+1}/{len(df)}...")

scores = np.array(scores)
print(f"\nScore range: {scores.min()} - {scores.max()}")
print(f"Mean: {scores.mean():.1f}")

bins   = [300,400,500,600,700,800,900]
labels = ["300-400","400-500","500-600","600-700","700-800","800-900"]
colors = ["#C0392B","#E67E22","#F1C40F","#27AE60","#1ABC9C","#2E86C1"]
xlabels= ["300-400\nVery Poor","400-500\nPoor","500-600\nFair",
          "600-700\nGood","700-800\nVery Good","800-900\nExcellent"]

counts = [int(np.sum((scores>=bins[i]) & (scores<bins[i+1])))
          for i in range(len(bins)-1)]
counts[-1] += int(np.sum(scores==900))
total  = sum(counts)
pcts   = [c/total*100 for c in counts]

print("\nRange breakdown:")
for l,c,p in zip(labels,counts,pcts):
    print(f"  {l}: {c:4d} ({p:.1f}%) {'█'*int(p/2)}")

fig, ax = plt.subplots(figsize=(10,6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
bars = ax.bar(xlabels, pcts, color=colors, width=0.6,
              edgecolor="white", linewidth=0.5)
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.4, f"{pct:.0f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#2C3E50")
ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="gray")
ax.set_axisbelow(True)
ax.set_title("Percentage of Users in Each Credit Score Range",
             fontsize=14, fontweight="bold", pad=15, color="#2C3E50")
ax.set_xlabel("Credit Score Range", fontsize=12, labelpad=10, color="#2C3E50")
ax.set_ylabel("Percentage of Users (%)", fontsize=12, labelpad=10, color="#2C3E50")
ax.set_ylim(0, max(pcts)*1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_alpha(0.3)
ax.spines["bottom"].set_alpha(0.3)
ax.tick_params(axis="both", labelsize=10, colors="#2C3E50")
plt.tight_layout()
os.makedirs("viz/output", exist_ok=True)
plt.savefig("viz/output/credit_score_distribution.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
