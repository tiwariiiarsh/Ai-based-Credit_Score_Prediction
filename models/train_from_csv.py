import os
import pickle
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# ---------------------------
# Load Dataset
# ---------------------------

df = pd.read_csv("data/abh.csv")

print("Rows loaded:", len(df))

# remove unwanted columns
drop_cols = [
    "customer_id",
    "profile_type",
    "state_tier",
    "age_bucket",
    "housing_type",
    "risk_segment",
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# features
feature_cols = [c for c in df.columns if c != "credit_score"]

# remove dominant feature for realistic model
if "payment_consistency_score" in feature_cols:
    feature_cols.remove("payment_consistency_score")

X = df[feature_cols]
y = df["credit_score"]

# ---------------------------
# Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Missing value handling
# ---------------------------

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ---------------------------
# Train Model
# ---------------------------

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n=== MODEL PERFORMANCE ===")
print("MAE:", round(mae, 2))
print("R2 :", round(r2, 4))

# ---------------------------
# Save Model
# ---------------------------

os.makedirs("models", exist_ok=True)

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "imputer": imputer,
        },
        f,
    )

print("\nModel saved → models/xgb_model.pkl")
