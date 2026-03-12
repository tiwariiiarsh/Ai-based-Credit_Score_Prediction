import pandas as pd

def build_user_features(users, monthly, loans, utilities):

    df = monthly.copy()

    # average income and spending
    agg = df.groupby("user_id").agg({
        "monthly_income": "mean",
        "monthly_spending": "mean",
        "total_transactions": "mean"
    }).reset_index()

    # spend-income ratio
    agg["spend_income_ratio"] = (
        agg["monthly_spending"] / (agg["monthly_income"] + 1)
    )

    # loan features
    loan_count = loans.groupby("user_id").size().reset_index(name="loan_count")

    agg = agg.merge(loan_count, on="user_id", how="left")

    agg["loan_count"] = agg["loan_count"].fillna(0)

    # utility payment behaviour
    utilities["on_time"] = (
        utilities["payment_date"] <= utilities["due_date"]
    ).astype(int)

    util_ratio = utilities.groupby("user_id")["on_time"].mean().reset_index()

    util_ratio.rename(
        columns={"on_time": "utility_payment_ratio"}, inplace=True
    )

    agg = agg.merge(util_ratio, on="user_id", how="left")

    agg["utility_payment_ratio"] = agg["utility_payment_ratio"].fillna(0)

    # add age
    agg = agg.merge(users[["user_id", "age"]], on="user_id", how="left")

    return agg