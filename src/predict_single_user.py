import joblib
import numpy as np
import pandas as pd

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
# USER ID
# -------------------------

USER_ID = "U0001"


# -------------------------
# FETCH DATA
# -------------------------

users, transactions, loans, utilities, monthly = fetch_data()

features = build_features(users, transactions, loans, utilities, monthly)


# -------------------------
# FILTER USER
# -------------------------

user_df = features[features["user_id"] == USER_ID]

if user_df.empty:

    print("User not found in database")

    exit()


X = user_df.drop(columns=["user_id"])

# ensure correct feature order
X = X[feature_cols]


# -------------------------
# CLUSTER
# -------------------------

X_scaled = scaler.transform(X)

cluster = gmm.predict(X_scaled)[0]


# -------------------------
# RISK PREDICTION
# -------------------------

risk_raw = model.predict(X)[0]

# sigmoid transform
risk = 1 / (1 + np.exp(-risk_raw))


# -------------------------
# CREDIT SCORE
# -------------------------

score = 300 + (1 - risk) * 600

score = int(np.clip(score, 300, 900))


# -------------------------
# RESULT
# -------------------------

print("\nSingle User Credit Score\n")

print("User ID:", USER_ID)
print("Cluster:", cluster)
print("Risk:", round(risk,3))
print("Credit Score:", score)