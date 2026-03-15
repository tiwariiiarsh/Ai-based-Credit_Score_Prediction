import numpy as np
import pandas as pd
import mysql.connector
from faker import Faker
from datetime import timedelta
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

conn = mysql.connector.connect(
    host="localhost", user="root",
    password="Arsh@1106", database="Barclays_Bank"
)
cursor = conn.cursor()

cursor.execute("SET FOREIGN_KEY_CHECKS=0")
for t in ["transactions","loans","utilities","user_monthly_features","users"]:
    cursor.execute(f"TRUNCATE TABLE {t}")
cursor.execute("SET FOREIGN_KEY_CHECKS=1")
conn.commit()

# EXACTLY CIBIL 2024-25 distribution
# 300-400: 4%  = 40
# 400-500: 12% = 120
# 500-600: 39% = 390  PEAK
# 600-700: 20% = 200
# 700-800: 16% = 160
# 800-900: 9%  = 90
# Total = 1000

SEGMENTS = [
    ("very_poor",  40),
    ("poor",      120),
    ("fair",      390),
    ("good",      200),
    ("very_good", 160),
    ("excellent",  90),
]

# Income and spending ranges per segment
# These directly control what score XGBoost will predict
INCOME_RANGE = {
    "very_poor":  (4000,   15000),
    "poor":       (12000,  28000),
    "fair":       (22000,  52000),
    "good":       (45000,  78000),
    "very_good":  (70000, 118000),
    "excellent":  (98000, 148000),
}
SPEND_RANGE = {
    "very_poor":  (1.20, 2.50),
    "poor":       (0.85, 1.25),
    "fair":       (0.55, 0.85),
    "good":       (0.38, 0.58),
    "very_good":  (0.22, 0.40),
    "excellent":  (0.10, 0.26),
}
DELAY_RANGE = {
    "very_poor":  (25,  60),
    "poor":       (10,  28),
    "fair":       (1,   10),
    "good":       (-2,   4),
    "very_good":  (-5,   1),
    "excellent":  (-8,  -1),
}
MISS_RATE = {
    "very_poor":  0.18,
    "poor":       0.12,
    "fair":       0.08,
    "good":       0.05,
    "very_good":  0.03,
    "excellent":  0.02,
}
LOAN_STATUS = {
    "very_poor":  ["missed"],
    "poor":       ["late","missed"],
    "fair":       ["on_time","late","late"],
    "good":       ["on_time","on_time","late"],
    "very_good":  ["on_time","on_time"],
    "excellent":  ["on_time"],
}

profiles = []
for seg, count in SEGMENTS:
    profiles.extend([seg]*count)
random.shuffle(profiles)

for i, seg in enumerate(profiles, start=1):
    uid = f"U{i:04d}"
    miss = MISS_RATE[seg]

    cursor.execute(
        "INSERT INTO users VALUES (%s,%s,%s,%s,%s)",
        (uid, random.randint(21,65),
         random.choice(["salaried","business","freelancer"]),
         random.choice(["owned","rented"]),
         random.choice(["tier1","tier2","tier3"]))
    )

    inc_lo, inc_hi = INCOME_RANGE[seg]
    spd_lo, spd_hi = SPEND_RANGE[seg]
    base_income = random.randint(inc_lo, inc_hi)
    volatility  = {"very_poor":0.35,"poor":0.25,"fair":0.18,
                   "good":0.12,"very_good":0.07,"excellent":0.03}[seg]

    missing_months = random.sample(range(36), k=random.randint(2,5))

    for m in range(36):
        year  = 2022 + m//12
        month = (m%12)+1
        income   = base_income * random.uniform(1-volatility, 1+volatility)
        spending = income * random.uniform(spd_lo, spd_hi)
        total_tx = random.randint(40,300)

        inc_val = None if (m in missing_months or random.random()<miss) else round(income,2)
        spd_val = None if random.random()<miss else round(spending,2)

        cursor.execute("""
            INSERT INTO user_monthly_features
            (user_id,year,month,total_transactions,
             monthly_spending,monthly_income,month_index)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (uid,year,month,total_tx,spd_val,inc_val,m))

    if seg in ["poor","fair","very_poor"] or random.random()<0.4:
        loan   = random.randint(20000,800000)
        tenure = random.randint(12,72)
        status = random.choice(LOAN_STATUS[seg])
        if random.random()<miss: status=None
        cursor.execute("""
            INSERT INTO loans (user_id,loan_amount,emi_amount,repayment_status)
            VALUES (%s,%s,%s,%s)
        """, (uid,loan,loan/tenure,status))

    dlo, dhi = DELAY_RANGE[seg]
    for _ in range(random.randint(30,80)):
        due     = fake.date_between(start_date="-5y", end_date="today")
        delay   = random.randint(dlo, dhi)
        payment = due + timedelta(days=delay)
        bill    = None if random.random()<miss else round(random.uniform(500,5000),2)
        cursor.execute("""
            INSERT INTO utilities (user_id,bill_amount,payment_date,due_date)
            VALUES (%s,%s,%s,%s)
        """, (uid,bill,payment,due))

    if i%100==0:
        print(f"  Generated {i}/1000...")
        conn.commit()

conn.commit()
conn.close()
print("Done! 1000 DB users with CIBIL distribution.")
