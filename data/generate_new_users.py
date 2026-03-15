"""
Generates 1000 new users CSV with CIBIL distribution
and 8% missing values — simulates real credit invisible users.
Same features as abh.csv.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# Load original dataset
df = pd.read_csv("data/abh.csv")
drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment","credit_score"]
df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = df_feat.columns.tolist()

# CIBIL 2024-25 distribution
TARGET = {
    (300, 400):  40,   # 4%
    (400, 500): 120,   # 12%
    (500, 600): 390,   # 39% peak
    (600, 700): 200,   # 20%
    (700, 800): 160,   # 16%
    (800, 900):  90,   # 9%
}

sampled = []
for (lo, hi), count in TARGET.items():
    seg = df[(df['credit_score'] >= lo) & (df['credit_score'] < hi)]
    sampled.append(
        seg.sample(count, replace=len(seg)<count, random_state=42)
    )

df_new = pd.concat(sampled).sample(frac=1, random_state=42)
df_new = df_new.drop(columns=[c for c in drop_cols if c in df_new.columns])
df_new = df_new[feature_cols].reset_index(drop=True)

# Add 8% missing values — simulate credit invisible users
np.random.seed(42)
for col in feature_cols:
    mask = np.random.random(len(df_new)) < 0.08
    df_new.loc[mask, col] = np.nan

print(f"New users: {len(df_new)} rows, {len(feature_cols)} features")
print(f"Missing values: {df_new.isnull().sum().sum()} ({df_new.isnull().sum().sum()/df_new.size*100:.1f}%)")
print("\nExpected CIBIL distribution:")
for (lo, hi), count in TARGET.items():
    pct = count/1000*100
    bar = "█" * int(pct/2)
    print(f"  {lo}-{hi}: {count} ({pct:.1f}%) {bar}")

df_new.to_csv("data/new_users.csv", index=False)
print("\nSaved → data/new_users.csv")
"""
Generates 1000 new users CSV with CIBIL distribution
and 8% missing values — simulates real credit invisible users.
Same features as abh.csv.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# Load original dataset
df = pd.read_csv("data/abh.csv")
drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment","credit_score"]
df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = df_feat.columns.tolist()

# CIBIL 2024-25 distribution
TARGET = {
    (300, 400):  40,   # 4%
    (400, 500): 120,   # 12%
    (500, 600): 390,   # 39% peak
    (600, 700): 200,   # 20%
    (700, 800): 160,   # 16%
    (800, 900):  90,   # 9%
}

sampled = []
for (lo, hi), count in TARGET.items():
    seg = df[(df['credit_score'] >= lo) & (df['credit_score'] < hi)]
    sampled.append(
        seg.sample(count, replace=len(seg)<count, random_state=42)
    )

df_new = pd.concat(sampled).sample(frac=1, random_state=42)
df_new = df_new.drop(columns=[c for c in drop_cols if c in df_new.columns])
df_new = df_new[feature_cols].reset_index(drop=True)

# Add 8% missing values — simulate credit invisible users
np.random.seed(42)
for col in feature_cols:
    mask = np.random.random(len(df_new)) < 0.08
    df_new.loc[mask, col] = np.nan

print(f"New users: {len(df_new)} rows, {len(feature_cols)} features")
print(f"Missing values: {df_new.isnull().sum().sum()} ({df_new.isnull().sum().sum()/df_new.size*100:.1f}%)")
print("\nExpected CIBIL distribution:")
for (lo, hi), count in TARGET.items():
    pct = count/1000*100
    bar = "█" * int(pct/2)
    print(f"  {lo}-{hi}: {count} ({pct:.1f}%) {bar}")

df_new.to_csv("data/new_users.csv", index=False)
print("\nSaved → data/new_users.csv")
"""
Generates 1000 new users CSV with CIBIL distribution
and 8% missing values — simulates real credit invisible users.
Same features as abh.csv.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# Load original dataset
df = pd.read_csv("data/abh.csv")
drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment","credit_score"]
df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = df_feat.columns.tolist()

# CIBIL 2024-25 distribution
TARGET = {
    (300, 400):  40,   # 4%
    (400, 500): 120,   # 12%
    (500, 600): 390,   # 39% peak
    (600, 700): 200,   # 20%
    (700, 800): 160,   # 16%
    (800, 900):  90,   # 9%
}

sampled = []
for (lo, hi), count in TARGET.items():
    seg = df[(df['credit_score'] >= lo) & (df['credit_score'] < hi)]
    sampled.append(
        seg.sample(count, replace=len(seg)<count, random_state=42)
    )

df_new = pd.concat(sampled).sample(frac=1, random_state=42)
df_new = df_new.drop(columns=[c for c in drop_cols if c in df_new.columns])
df_new = df_new[feature_cols].reset_index(drop=True)

# Add 8% missing values — simulate credit invisible users
np.random.seed(42)
for col in feature_cols:
    mask = np.random.random(len(df_new)) < 0.08
    df_new.loc[mask, col] = np.nan

print(f"New users: {len(df_new)} rows, {len(feature_cols)} features")
print(f"Missing values: {df_new.isnull().sum().sum()} ({df_new.isnull().sum().sum()/df_new.size*100:.1f}%)")
print("\nExpected CIBIL distribution:")
for (lo, hi), count in TARGET.items():
    pct = count/1000*100
    bar = "█" * int(pct/2)
    print(f"  {lo}-{hi}: {count} ({pct:.1f}%) {bar}")

df_new.to_csv("data/new_users.csv", index=False)
print("\nSaved → data/new_users.csv")
"""
Generates 1000 new users CSV with CIBIL distribution
and 8% missing values — simulates real credit invisible users.
Same features as abh.csv.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# Load original dataset
df = pd.read_csv("data/abh.csv")
drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment","credit_score"]
df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = df_feat.columns.tolist()

# CIBIL 2024-25 distribution
TARGET = {
    (300, 400):  40,   # 4%
    (400, 500): 120,   # 12%
    (500, 600): 390,   # 39% peak
    (600, 700): 200,   # 20%
    (700, 800): 160,   # 16%
    (800, 900):  90,   # 9%
}

sampled = []
for (lo, hi), count in TARGET.items():
    seg = df[(df['credit_score'] >= lo) & (df['credit_score'] < hi)]
    sampled.append(
        seg.sample(count, replace=len(seg)<count, random_state=42)
    )

df_new = pd.concat(sampled).sample(frac=1, random_state=42)
df_new = df_new.drop(columns=[c for c in drop_cols if c in df_new.columns])
df_new = df_new[feature_cols].reset_index(drop=True)

# Add 8% missing values — simulate credit invisible users
np.random.seed(42)
for col in feature_cols:
    mask = np.random.random(len(df_new)) < 0.08
    df_new.loc[mask, col] = np.nan

print(f"New users: {len(df_new)} rows, {len(feature_cols)} features")
print(f"Missing values: {df_new.isnull().sum().sum()} ({df_new.isnull().sum().sum()/df_new.size*100:.1f}%)")
print("\nExpected CIBIL distribution:")
for (lo, hi), count in TARGET.items():
    pct = count/1000*100
    bar = "█" * int(pct/2)
    print(f"  {lo}-{hi}: {count} ({pct:.1f}%) {bar}")

df_new.to_csv("data/new_users.csv", index=False)
print("\nSaved → data/new_users.csv")
