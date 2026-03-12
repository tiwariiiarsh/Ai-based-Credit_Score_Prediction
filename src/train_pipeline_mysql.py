from src.build_training_data_mysql import fetch_data
from src.feature_engineering_mysql import build_user_features
from src.train_gmm_mysql import train_gmm
from src.train_xgboost_mysql import train_model

users, monthly, loans, utilities = fetch_data()

features = build_user_features(users, monthly, loans, utilities)

clustered = train_gmm(features)

train_model(clustered)

print("Training complete")