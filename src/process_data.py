import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/data_Set.csv"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fix numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace impossible negative values for amount-like columns
    amount_cols = [
        "monthly_income", "monthly_spending", "utility_bill_amount",
        "monthly_sales", "income_std_6m"
    ]
    for col in amount_cols:
        if col in df.columns:
            df[col] = df[col].abs()

    # Fill missing categorical
    cat_cols = ["employment_type", "residence_type", "city_tier"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()

    # Fill missing numeric with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def add_month_level_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eps = 1e-6

    # Safe ratios
    df["spend_income_ratio"] = df["monthly_spending"] / (df["monthly_income"] + eps)
    df["savings_amount"] = df["monthly_income"] - df["monthly_spending"]
    df["savings_ratio"] = df["savings_amount"] / (df["monthly_income"] + eps)

    df["utility_discipline_index"] = (
        0.5 * df["utility_payment_ratio"] +
        0.3 * df["bill_payment_consistency"] -
        0.2 * (df["utility_delay_days"] / 30.0)
    )

    df["credit_proxy_index"] = (
        0.4 * df["bnpl_on_time_ratio"] +
        0.3 * df["microloan_repayment_ratio"] -
        0.15 * df["bnpl_usage_count"] / 10.0 -
        0.15 * df["microloan_count"] / 10.0
    )

    df["stability_index"] = (
        0.35 * (1 / (1 + df["income_std_6m"])) +
        0.25 * (1 / (1 + df["salary_day_variance"])) +
        0.20 * (1 / (1 + df["spending_volatility"])) +
        0.20 * (1 - np.clip(df["financial_stress_index"], 0, 1))
    )

    df["payment_risk_index"] = (
        0.4 * df["late_payment_events"] +
        0.4 * df["missed_payment_events"] +
        0.2 * df["utility_delay_days"]
    )

    df["business_health_index"] = (
        0.35 * df["sales_growth_rate"] +
        0.25 * (1 / (1 + df["cashflow_variance"])) +
        0.20 * (1 - df["business_expense_ratio"]) +
        0.20 * (df["monthly_sales"] / (df["monthly_income"] + eps))
    )

    # Clip some ratios
    ratio_cols = [
        "spend_income_ratio", "savings_ratio",
        "utility_payment_ratio", "bill_payment_consistency",
        "bnpl_on_time_ratio", "microloan_repayment_ratio"
    ]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].clip(-5, 5)

    return df


def aggregate_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sort for consistent month operations
    df = df.sort_values(["user_id", "year", "month"])

    # Add per-user trend features
    def compute_user_trend(group: pd.DataFrame) -> pd.Series:
        income_trend = group["monthly_income"].iloc[-1] - group["monthly_income"].iloc[0]
        spending_trend = group["monthly_spending"].iloc[-1] - group["monthly_spending"].iloc[0]
        utility_avg_delay = group["utility_delay_days"].mean()
        ontime_behavior = (
            0.5 * group["bnpl_on_time_ratio"].mean() +
            0.5 * group["microloan_repayment_ratio"].mean()
        )

        return pd.Series({
            "income_trend_12m": income_trend,
            "spending_trend_12m": spending_trend,
            "utility_avg_delay_12m": utility_avg_delay,
            "repayment_behavior_12m": ontime_behavior
        })

    trend_df = df.groupby("user_id").apply(compute_user_trend).reset_index()

    # One-hot encode categorical before aggregation
    df_encoded = pd.get_dummies(
        df,
        columns=["employment_type", "residence_type", "city_tier"],
        drop_first=False
    )

    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["year", "month"]]

    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ["mean", "std", "min", "max"]

    user_features = df_encoded.groupby("user_id").agg(agg_dict)
    user_features.columns = [
        f"{col}_{stat}" for col, stat in user_features.columns
    ]
    user_features = user_features.reset_index()

    # Fill std NaNs with 0
    user_features = user_features.fillna(0)

    # Merge trends
    user_features = user_features.merge(trend_df, on="user_id", how="left")

    # Extra final engineered user-level features
    eps = 1e-6
    if "monthly_income_mean" in user_features.columns and "monthly_spending_mean" in user_features.columns:
        user_features["avg_surplus"] = (
            user_features["monthly_income_mean"] - user_features["monthly_spending_mean"]
        )
        user_features["avg_surplus_ratio"] = (
            user_features["avg_surplus"] / (user_features["monthly_income_mean"] + eps)
        )

    if "late_payment_events_mean" in user_features.columns and "missed_payment_events_mean" in user_features.columns:
        user_features["total_payment_problem_index"] = (
            user_features["late_payment_events_mean"] +
            user_features["missed_payment_events_mean"]
        )

    return user_features


def fit_and_scale_features(user_features: pd.DataFrame):
    X = user_features.drop(columns=["user_id"]).copy()

    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    return user_features, X, X_scaled, scaler, feature_columns


def transform_new_user_features(user_features: pd.DataFrame):
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    X = user_features.copy()

    # Add missing columns as 0
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Remove extra columns
    X = X[feature_columns]

    X_scaled = scaler.transform(X)
    return X, X_scaled


def process_training_data():
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    df = add_month_level_features(df)
    user_features = aggregate_user_features(df)
    user_features, X, X_scaled, scaler, feature_columns = fit_and_scale_features(user_features)
    return df, user_features, X, X_scaled, scaler, feature_columns


if __name__ == "__main__":
    df, user_features, X, X_scaled, scaler, feature_columns = process_training_data()
    print("Raw monthly shape:", df.shape)
    print("User-level shape:", user_features.shape)
    print(user_features.head())