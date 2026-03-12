import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import random
from src.predict_credit import predict_credit_score_from_df

df = pd.read_csv("data/data_Set.csv")

users = df["user_id"].unique()

random_user = random.choice(users)

print("Testing user:", random_user)

user_df = df[df["user_id"] == random_user]

result = predict_credit_score_from_df(user_df)

print(result)