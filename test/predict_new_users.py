import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.predict_credit import predict_credit_score_from_df

print("Loading new users dataset...")

df = pd.read_csv("data/new_users.csv")

users = df["user_id"].unique()

results = []

print("Total users:", len(users))

for uid in users:

    user_df = df[df["user_id"] == uid]

    result = predict_credit_score_from_df(user_df)

    results.append({
        "user_id": result["user_id"],
        "risk_score": result["risk_score"],
        "credit_score": result["credit_score"],
        "risk_band": result["risk_band"]
    })

scores = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

scores.to_csv("outputs/new_user_scores.csv", index=False)

print("\nPrediction complete!")
print("Saved to outputs/new_user_scores.csv")
print(scores.head())