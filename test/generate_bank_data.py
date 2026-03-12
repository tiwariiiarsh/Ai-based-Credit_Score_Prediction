import random
import mysql.connector
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Arsh@1106",
    database="Barclays_Bank"
)

cursor = conn.cursor()

employment = ["salaried","business","freelancer","self-employed"]
residence = ["owned","rented","family"]
cities = ["tier1","tier2","tier3"]

categories = [
"grocery","electronics","fuel",
"shopping","medical","travel"
]

# -------- USERS --------

for i in range(1,501):

    user_id = f"U{i:04d}"

    age = random.randint(21,65)
    emp = random.choice(employment)
    res = random.choice(residence)
    city = random.choice(cities)

    cursor.execute(
    "INSERT INTO users VALUES (%s,%s,%s,%s,%s)",
    (user_id,age,emp,res,city)
    )

    # -------- TRANSACTIONS --------

    start_date = datetime(2018,1,1)

    for _ in range(random.randint(200,500)):

        date = start_date + timedelta(days=random.randint(0,2000))

        amount = random.uniform(100,20000)

        ttype = random.choice(["debit","credit"])
        category = random.choice(categories)

        cursor.execute("""
        INSERT INTO transactions
        (user_id,date,transaction_amount,transaction_type,merchant_category)
        VALUES (%s,%s,%s,%s,%s)
        """,(user_id,date,amount,ttype,category))


    # -------- LOANS --------

    if random.random() < 0.5:

        loan = random.randint(50000,500000)
        emi = loan/random.randint(12,60)
        status = random.choice(["on_time","late","missed"])

        cursor.execute("""
        INSERT INTO loans
        (user_id,loan_amount,emi_amount,repayment_status)
        VALUES (%s,%s,%s,%s)
        """,(user_id,loan,emi,status))


    # -------- UTILITIES --------

    for _ in range(random.randint(30,80)):

        due = fake.date_between(start_date="-5y",end_date="today")
        payment = due + timedelta(days=random.randint(-2,10))

        bill = random.uniform(500,5000)

        cursor.execute("""
        INSERT INTO utilities
        (user_id,bill_amount,payment_date,due_date)
        VALUES (%s,%s,%s,%s)
        """,(user_id,bill,payment,due))


conn.commit()

print("Dummy Bank Data Generated")