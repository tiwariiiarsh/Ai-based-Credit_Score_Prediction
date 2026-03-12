import pandas as pd
import mysql.connector

def fetch_data():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Arsh@1106",
        database="Barclays_Bank"
    )

    users = pd.read_sql("SELECT * FROM users", conn)
    transactions = pd.read_sql("SELECT * FROM transactions", conn)
    loans = pd.read_sql("SELECT * FROM loans", conn)
    utilities = pd.read_sql("SELECT * FROM utilities", conn)
    monthly = pd.read_sql("SELECT * FROM user_monthly_features", conn)

    conn.close()

    return users, transactions, loans, utilities, monthly