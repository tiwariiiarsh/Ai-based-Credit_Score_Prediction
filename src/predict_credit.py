import joblib
import numpy as np
import pandas as pd

from src.process_data import (
    clean_data,
    add_month_level_features,
    aggregate_user_features,
    transform_new_user_features
)

GMM_PATH = "models/gmm_model.pkl"
XGB_PATH = "models/xgboost_model.pkl"
CLUSTER_MEANS_PATH = "models/cluster_means.csv"


def build_new_user_feature_row(new_user_monthly_df: pd.DataFrame):

    df = clean_data(new_user_monthly_df)
    df = add_month_level_features(df)
    user_features = aggregate_user_features(df)

    X_user = user_features.drop(columns=["user_id"]).copy()

    return user_features[["user_id"]], X_user


def impute_missing_with_gmm(X_user: pd.DataFrame):

    gmm = joblib.load(GMM_PATH)
    cluster_means = pd.read_csv(CLUSTER_MEANS_PATH, index_col=0)

    X_user_imputed = X_user.copy()

    # temporary fill for cluster membership
    temp_user = X_user_imputed.copy()

    for col in temp_user.columns:
        if temp_user[col].isnull().any():
            if col in cluster_means.columns:
                temp_user[col] = temp_user[col].fillna(cluster_means[col].mean())
            else:
                temp_user[col] = temp_user[col].fillna(0)

    _, temp_scaled = transform_new_user_features(temp_user)

    probs = gmm.predict_proba(temp_scaled)[0]

    # weighted imputation
    for col in X_user_imputed.columns:
        if X_user_imputed[col].isnull().any():

            if col in cluster_means.columns:
                weighted_value = np.sum(probs * cluster_means[col].values)
                X_user_imputed[col] = X_user_imputed[col].fillna(weighted_value)

            else:
                X_user_imputed[col] = X_user_imputed[col].fillna(0)

    return X_user_imputed, probs


# -------- NEW CREDIT SCORE LOGIC --------

def risk_to_credit_score(risk: float) -> int:
    """
    risk range: 0 → low risk
    risk range: 1 → high risk
    """

    risk = np.clip(risk, 0, 1)

    score = 900 - risk * 600

    return int(round(score))


def risk_band(score: int):

    if score >= 750:
        return "A - Auto Approve"

    elif score >= 650:
        return "B - Approve with Lower Limit"

    elif score >= 550:
        return "C - Manual Review / Collateral"

    else:
        return "D - Reject"


def predict_credit_score_from_df(new_user_monthly_df: pd.DataFrame):

    xgb_model = joblib.load(XGB_PATH)

    user_info, X_user = build_new_user_feature_row(new_user_monthly_df)

    # handle missing values
    X_user_imputed, cluster_probs = impute_missing_with_gmm(X_user)

    _, X_scaled = transform_new_user_features(X_user_imputed)

    # predict continuous risk
    risk_score = float(xgb_model.predict(X_scaled)[0])

    credit_score = risk_to_credit_score(risk_score)

    band = risk_band(credit_score)

    result = {
        "user_id": user_info.iloc[0]["user_id"],
        "risk_score": round(risk_score, 4),
        "credit_score": credit_score,
        "risk_band": band,
        "cluster_probabilities": {
            f"cluster_{i}": round(float(p), 4)
            for i, p in enumerate(cluster_probs)
        }
    }

    return result


if __name__ == "__main__":

    raw_df = pd.read_csv("data/data_Set.csv")

    sample_user_df = raw_df[
        raw_df["user_id"] == raw_df["user_id"].iloc[0]
    ].copy()

    # artificial missing values
    sample_user_df.loc[sample_user_df.index[:3], "monthly_income"] = np.nan
    sample_user_df.loc[sample_user_df.index[:2], "bnpl_on_time_ratio"] = np.nan

    result = predict_credit_score_from_df(sample_user_df)

    print(result)