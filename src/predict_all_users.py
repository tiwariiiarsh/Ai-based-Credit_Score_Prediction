import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.fetch_mysql_data import fetch_data
from src.feature_engineering_mysql import build_features

# -------------------------
# LOAD MODELS
# -------------------------

gmm = joblib.load("models/gmm_model.pkl")
model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")


# -------------------------
# FETCH DATA
# -------------------------

users, transactions, loans, utilities, monthly = fetch_data()

features = build_features(users, transactions, loans, utilities, monthly)

X = features.drop(columns=["user_id"])

# ensure correct feature order
X = X[feature_cols]


# -------------------------
# CLUSTER PREDICTION
# -------------------------

X_scaled = scaler.transform(X)

clusters = gmm.predict(X_scaled)


# -------------------------
# RISK PREDICTION
# -------------------------

risk_raw = model.predict(X)

# sigmoid transform (keeps risk between 0-1)
risk = 1 / (1 + np.exp(-risk_raw))

print("Risk min:", risk.min())
print("Risk max:", risk.max())
print("Risk mean:", risk.mean())


# -------------------------
# CREDIT SCORE
# -------------------------

scores = 300 + (1 - risk) * 600

scores = np.clip(scores, 300, 900)


# -------------------------
# SAVE RESULTS
# -------------------------

features["cluster"] = clusters
features["credit_score"] = scores.astype(int)


print("\nPrediction Sample\n")

print(features[["user_id","cluster","credit_score"]].head())


# -------------------------
# GRAPH
# -------------------------

plt.figure()

features["credit_score"].hist(bins=20)

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Users")

plt.show()