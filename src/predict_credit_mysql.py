import joblib
import numpy as np
from src.fetch_mysql_data import fetch_data
from src.feature_engineering_mysql import build_features

gmm = joblib.load("models/gmm_model.pkl")
model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

users, transactions, loans, utilities, monthly = fetch_data()

features = build_features(users, transactions, loans, utilities, monthly)

X = features.drop(columns=["user_id"])

X = X[feature_cols]

X_scaled = scaler.transform(X)

clusters = gmm.predict(X_scaled)

risk = model.predict(X)

# credit score
scores = 300 + (1 - risk) * 600

scores = np.clip(scores,300,900)

features["cluster"] = clusters
features["credit_score"] = scores.astype(int)

print(features[["user_id","cluster","credit_score"]].head())