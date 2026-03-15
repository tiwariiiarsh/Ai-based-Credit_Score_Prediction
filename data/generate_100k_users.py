"""
Generate 100,000 synthetic users dataset
Same feature pattern as abh.csv
Maintains CIBIL distribution
Adds 8% missing values
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Load training dataset
df = pd.read_csv("data/abh.csv")

# Drop non-feature columns
drop_cols = [
    "customer_id",
    "profile_type",
    "state_tier",
    "age_bucket",
    "housing_type",
    "risk_segment",
    "credit_score"
]

df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = df_feat.columns.tolist()

# 100K CIBIL distribution
TARGET = {
    (300, 400): 4000,
    (400, 500): 12000,
    (500, 600): 39000,
    (600, 700): 20000,
    (700, 800): 16000,
    (800, 900): 9000,
}

sampled = []

for (lo, hi), count in TARGET.items():

    seg = df[(df["credit_score"] >= lo) & (df["credit_score"] < hi)]

    sampled.append(
        seg.sample(
            count,
            replace=len(seg) < count,
            random_state=42
        )
    )

df_new = pd.concat(sampled)

# shuffle rows
df_new = df_new.sample(frac=1, random_state=42)

# drop non-feature columns
df_new = df_new.drop(columns=[c for c in drop_cols if c in df_new.columns])

# keep feature order same as training
df_new = df_new[feature_cols].reset_index(drop=True)

# add 8% missing values
np.random.seed(42)

for col in feature_cols:

    mask = np.random.random(len(df_new)) < 0.08
    df_new.loc[mask, col] = np.nan

# save dataset
df_new.to_csv("data/new_users_100k.csv", index=False)

print("Dataset generated successfully\n")

print("Rows:", len(df_new))
print("Features:", len(feature_cols))

missing = df_new.isnull().sum().sum()

print("Missing values:", missing,
      f"({missing/df_new.size*100:.1f}%)")

print("\nExpected CIBIL distribution:")

for (lo, hi), count in TARGET.items():

    pct = count / 100000 * 100
    print(f"{lo}-{hi}: {count} ({pct:.1f}%)")

print("\nSaved → data/new_users_100k.csv")