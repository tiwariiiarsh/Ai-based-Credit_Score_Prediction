import joblib
from src.build_training_data_mysql import fetch_data
from src.feature_engineering_mysql import build_user_features

users, monthly, loans, utilities = fetch_data()

features = build_user_features(users, monthly, loans, utilities)

model = joblib.load("models/xgboost_model.pkl")

X = features.drop(columns=["user_id"])

pred = model.predict(X)

features["credit_cluster"] = pred

print(features.head())