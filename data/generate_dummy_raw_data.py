import pandas as pd
import numpy as np

users = []
monthly = []

n_users = 100
months = 120

for i in range(n_users):

    user_id = f"U{i:04d}"

    age = np.random.randint(21,60)

    users.append({
        "user_id":user_id,
        "age":age,
        "employment_type":"salaried",
        "residence_type":"rent",
        "city":"tier2"
    })

    for m in range(months):

        monthly.append({
            "user_id":user_id,
            "year":2015 + m//12,
            "month":m%12 + 1,
            "monthly_income":np.random.normal(30000,8000),
            "monthly_spending":np.random.normal(20000,6000),
            "total_transactions":np.random.randint(10,80)
        })


users_df = pd.DataFrame(users)
monthly_df = pd.DataFrame(monthly)

users_df.to_csv("data/users.csv",index=False)
monthly_df.to_csv("data/user_monthly_features.csv",index=False)

print("Dummy raw dataset generated")


# ye files milengi
# data/users.csv
# data/user_monthly_features.csv