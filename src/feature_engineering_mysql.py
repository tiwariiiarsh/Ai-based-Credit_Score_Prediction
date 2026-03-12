import pandas as pd
import numpy as np


def build_features(users, transactions, loans, utilities, monthly):

    # -------------------------
    # INCOME FEATURES
    # -------------------------

    income = monthly.groupby("user_id")["monthly_income"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()

    income.columns = [
        "user_id",
        "avg_income",
        "income_volatility",
        "min_income",
        "max_income"
    ]

    income["income_stability"] = 1 / (income["income_volatility"] + 1)


    # -------------------------
    # SPENDING FEATURES
    # -------------------------

    spending = monthly.groupby("user_id")["monthly_spending"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()

    spending.columns = [
        "user_id",
        "avg_spending",
        "spending_volatility",
        "min_spending",
        "max_spending"
    ]

    spending["spending_stability"] = 1 / (spending["spending_volatility"] + 1)

    spending["spend_income_ratio"] = (
        spending["avg_spending"] /
        (income["avg_income"] + 1)
    )


    # -------------------------
    # TRANSACTION FEATURES
    # -------------------------

    tx_count = transactions.groupby("user_id").size().reset_index(
        name="transaction_count"
    )

    avg_tx_amount = transactions.groupby("user_id")["transaction_amount"].mean().reset_index(
        name="avg_transaction_amount"
    )

    tx_std = transactions.groupby("user_id")["transaction_amount"].std().reset_index(
        name="transaction_amount_std"
    )

    debit_tx = transactions[
        transactions["transaction_type"] == "debit"
    ].groupby("user_id").size()

    total_tx = transactions.groupby("user_id").size()

    debit_ratio = (debit_tx / total_tx).fillna(0).reset_index(
        name="debit_ratio"
    )

    transaction_frequency = total_tx.reset_index(
        name="transaction_frequency"
    )


    # -------------------------
    # LOAN FEATURES
    # -------------------------

    loan_count = loans.groupby("user_id").size().reset_index(
        name="loan_count"
    )

    avg_loan = loans.groupby("user_id")["loan_amount"].mean().reset_index(
        name="avg_loan_amount"
    )

    emi_mean = loans.groupby("user_id")["emi_amount"].mean().reset_index(
        name="avg_emi"
    )

    loan_pressure = loans.groupby("user_id")["emi_amount"].sum().reset_index(
        name="total_emi_pressure"
    )


    # -------------------------
    # UTILITY FEATURES
    # -------------------------

    utilities["payment_date"] = pd.to_datetime(utilities["payment_date"], errors="coerce")
    utilities["due_date"] = pd.to_datetime(utilities["due_date"], errors="coerce")

    utilities["on_time"] = (
        utilities["payment_date"] <= utilities["due_date"]
    ).astype(int)

    utility_ratio = utilities.groupby("user_id")["on_time"].mean().reset_index(
        name="utility_payment_ratio"
    )

    avg_bill = utilities.groupby("user_id")["bill_amount"].mean().reset_index(
        name="avg_bill_amount"
    )

    late_payments = utilities[
        utilities["payment_date"] > utilities["due_date"]
    ].groupby("user_id").size().reset_index(
        name="late_payment_count"
    )

    utilities["delay_days"] = (
        utilities["payment_date"] - utilities["due_date"]
    ).dt.days.clip(lower=0)

    avg_delay = utilities.groupby("user_id")["delay_days"].mean().reset_index(
        name="avg_utility_delay")


    # -------------------------
    # MERGE ALL FEATURES
    # -------------------------

    df = income.merge(spending, on="user_id")

    df = df.merge(tx_count, on="user_id", how="left")
    df = df.merge(avg_tx_amount, on="user_id", how="left")
    df = df.merge(tx_std, on="user_id", how="left")
    df = df.merge(debit_ratio, on="user_id", how="left")
    df = df.merge(transaction_frequency, on="user_id", how="left")

    df = df.merge(loan_count, on="user_id", how="left")
    df = df.merge(avg_loan, on="user_id", how="left")
    df = df.merge(emi_mean, on="user_id", how="left")
    df = df.merge(loan_pressure, on="user_id", how="left")

    df = df.merge(utility_ratio, on="user_id", how="left")
    df = df.merge(avg_bill, on="user_id", how="left")
    df = df.merge(late_payments, on="user_id", how="left")
    df = df.merge(avg_delay, on="user_id", how="left")

    df = df.merge(users[["user_id", "age"]], on="user_id")


    # -------------------------
    # DERIVED FEATURES
    # -------------------------

    df["loan_income_ratio"] = df["avg_loan_amount"] / (df["avg_income"] + 1)

    df["emi_income_ratio"] = df["avg_emi"] / (df["avg_income"] + 1)

    df["financial_stress_index"] = (
        df["spend_income_ratio"] * 0.4 +
        df["loan_income_ratio"] * 0.2 +
        (1 - df["utility_payment_ratio"]) * 0.4
    )


    # -------------------------
    # CLEANUP
    # -------------------------

    df = df.fillna(0)

    return df