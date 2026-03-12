import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.process_data import process_training_data

XGB_PATH = "models/xgboost_model.pkl"


def create_proxy_target(user_features: pd.DataFrame) -> pd.Series:
    uf = user_features.copy()

    # Behaviour based risk score
    risk_score = (
        0.20 * uf.get("spend_income_ratio_mean", 0) +
        0.15 * uf.get("financial_stress_index_mean", 0) +
        0.15 * uf.get("late_payment_events_mean", 0) +
        0.15 * uf.get("missed_payment_events_mean", 0) +
        0.10 * uf.get("utility_delay_days_mean", 0) +
        0.10 * (1 - uf.get("utility_payment_ratio_mean", 0)) +
        0.10 * (1 - uf.get("bnpl_on_time_ratio_mean", 0)) +
        0.05 * (1 - uf.get("microloan_repayment_ratio_mean", 0))
    )

    # Normalize between 0–1
    risk_score = (risk_score - risk_score.min()) / (
        risk_score.max() - risk_score.min() + 1e-6
    )

    return risk_score


def train_xgboost():

    # Load processed data
    _, user_features, X, X_scaled, _, _ = process_training_data()

    # Continuous risk target
    y = create_proxy_target(user_features)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Regression model
    model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, XGB_PATH)

    print("XGBoost Risk Regression model trained successfully")
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))
    print("Model saved to:", XGB_PATH)


if __name__ == "__main__":
    train_xgboost()






# import os
# import joblib
# import numpy as np
# import pandas as pd
# import xgboost as xgb

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score

# from src.process_data import process_training_data

# XGB_PATH = "models/xgboost_model.pkl"


# def create_proxy_target(user_features: pd.DataFrame) -> pd.Series:
#     uf = user_features.copy()

#     # Risk score bana rahe hain based on behavior
#     risk_score = (
#         0.20 * uf.get("spend_income_ratio_mean", 0) +
#         0.15 * uf.get("financial_stress_index_mean", 0) +
#         0.15 * uf.get("late_payment_events_mean", 0) +
#         0.15 * uf.get("missed_payment_events_mean", 0) +
#         0.10 * uf.get("utility_delay_days_mean", 0) +
#         0.10 * (1 - uf.get("utility_payment_ratio_mean", 0)) +
#         0.10 * (1 - uf.get("bnpl_on_time_ratio_mean", 0)) +
#         0.05 * (1 - uf.get("microloan_repayment_ratio_mean", 0))
#     )

#     # Normalize 0-1
#     risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-6)

#     # Binary target
#     y = (risk_score > risk_score.median()).astype(int)
#     return y


# def train_xgboost():
#     _, user_features, X, X_scaled, _, _ = process_training_data()

#     y = create_proxy_target(user_features)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42, stratify=y
#     )

#     model = xgb.XGBClassifier(
#         n_estimators=250,
#         max_depth=5,
#         learning_rate=0.05,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         objective="binary:logistic",
#         eval_metric="logloss",
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     acc = accuracy_score(y_test, y_pred)
#     try:
#         auc = roc_auc_score(y_test, y_prob)
#     except Exception:
#         auc = None

#     os.makedirs("models", exist_ok=True)
#     joblib.dump(model, XGB_PATH)

#     print("XGBoost model trained successfully")
#     print("Accuracy:", round(acc, 4))
#     if auc is not None:
#         print("AUC:", round(auc, 4))
#     print("Model saved to:", XGB_PATH)


# if __name__ == "__main__":
#     train_xgboost()