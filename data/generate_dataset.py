import pandas as pd
import numpy as np
import random

users = 1000
months = 36

data = []

for u in range(1, users+1):

    user_id = f"U{u:04d}"

    age = random.randint(21,60)

    employment_type = random.choice([
        "salaried","self_employed","business"
    ])

    residence_type = random.choice(["owned","rented"])

    city_tier = random.choice(["tier1","tier2","tier3"])

    base_income = random.randint(15000,120000)

    for m in range(1, months+1):

        income = base_income + np.random.normal(0,8000)

        spending = income * random.uniform(0.25,0.9)

        row = {

        "user_id":user_id,
        "year":2025,
        "month":m,

        "age":age,
        "employment_type":employment_type,
        "residence_type":residence_type,
        "city_tier":city_tier,

        "monthly_income":round(income,2),
        "income_sources_count":random.randint(1,4),
        "income_std_6m":abs(np.random.normal(5000,2000)),
        "salary_day_variance":random.randint(0,7),

        "monthly_spending":round(spending,2),
        "spend_income_ratio":round(spending/income,2),
        "discretionary_spend_ratio":round(random.uniform(0.2,0.6),2),
        "essential_spend_ratio":round(random.uniform(0.4,0.8),2),
        "spending_volatility":round(random.uniform(0.05,0.35),2),

        "balance_low_days":random.randint(0,12),

        "utility_bill_amount":random.randint(800,6000),
        "utility_payment_ratio":round(random.uniform(0.6,1),2),
        "utility_delay_days":random.randint(0,8),
        "bill_payment_consistency":round(random.uniform(0.5,1),2),

        "bnpl_usage_count":random.randint(0,6),
        "bnpl_on_time_ratio":round(random.uniform(0.5,1),2),

        "microloan_count":random.randint(0,4),
        "microloan_repayment_ratio":round(random.uniform(0.5,1),2),

        "monthly_sales":random.randint(0,250000) if employment_type=="business" else 0,
        "sales_growth_rate":round(random.uniform(-0.2,0.4),2),
        "cashflow_variance":round(random.uniform(0.1,0.6),2),
        "business_expense_ratio":round(random.uniform(0.3,0.8),2),

        "late_payment_events":random.randint(0,4),
        "missed_payment_events":random.randint(0,3),

        "financial_stress_index":round(random.uniform(0.1,1),2)

        }

        data.append(row)

df = pd.DataFrame(data)

df.to_csv("data/data_Set.csv",index=False)

print("Dataset generated:",df.shape)