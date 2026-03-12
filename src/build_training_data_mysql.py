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
    monthly = pd.read_sql("SELECT * FROM user_monthly_features", conn)
    loans = pd.read_sql("SELECT * FROM loans", conn)
    utilities = pd.read_sql("SELECT * FROM utilities", conn)

    conn.close()

    return users, monthly, loans, utilities