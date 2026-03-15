import pickle
import numpy as np
import pandas as pd
import mysql.connector
from scoring.impute import load_gmm, impute_features

def get_connection():
    return mysql.connector.connect(
        host="localhost", user="root",
        passwordimport pickle
import numpy as np
import pandas as pd
import mysql.connector
from scoring.impute import load_gmm, impute_features

def get_connection():
    return mysql.connector.connect(
        host="localhost", user="root",
        password="Arsh@1106", database="Barclays_Bank"
    )

def assign_band(score):
    if score >= 750: return "A", "Auto-Approve"
    if score >= 620: return "B", "Approve with lower limit"
    if score >= 500: return "C", "Require additional collateral / review"
    return "D", "Reject"

def fetch_user_features(user_id: str, cursor) -> dict:
    cursor.execute(
        "SELECT monthly_income, monthly_spending, total_transactions "
        "FROM user_monthly_features WHERE user_id=%s ORDER BY month_index",
        (user_id,)
    )
    rows = cursor.fetchall()
    if not rows:
        return {}

    inc = [r[0] for r in rows if r[0] is not None]
    spd = [r[1] for r in rows if r[1] is not None]
    txn = [r[2] for r in rows if r[2] is not None]

    def safe_cv(arr):
        return float(np.std(arr)/np.mean(arr)) if arr and np.mean(arr)!=0 else np.nan

    f = {}
    avg_inc = float(np.mean(inc)) if inc else np.nan
    avg_spd = float(np.mean(spd)) if spd else np.nan
    sr      = [s/i for s,i in zip(spd,inc) if i and i>0]

    f["avg_monthly_income"]        = avg_inc
    f["income_std_dev"]            = float(np.std(inc)) if inc else np.nan
    f["income_trend"]              = float((inc[-1]-inc[0])/inc[0]) if len(inc)>1 and inc[0] else np.nan
    f["income_sources_count"]      = 1.0
    f["salary_days_variance"]      = float(np.clip(np.var(inc)/1e8, 0, 20)) if inc else np.nan
    f["avg_monthly_spend"]         = avg_spd
    f["spend_to_income_ratio"]     = float(np.clip(np.mean(sr), 0.1, 2.0)) if sr else np.nan
    f["discretionary_spend_ratio"] = float(np.clip(np.mean(sr)*0.5, 0.04, 0.8)) if sr else np.nan
    f["balance_low_days"]          = float(sum(1 for r in sr if r>0.9))
    f["cash_withdrawal_ratio"]     = 0.33

    cursor.execute(
        "SELECT bill_amount, payment_date, due_date FROM utilities WHERE user_id=%s",
        (user_id,)
    )
    util_rows = cursor.fetchall()
    bills, delays = [], []
    for r in util_rows:
        if r[0]: bills.append(float(r[0]))
        if r[1] and r[2]:
            delays.append((pd.to_datetime(r[1])-pd.to_datetime(r[2])).days)

    f["utility_payment_ratio"]      = float(sum(1 for d in delays if d<=0)/len(delays)) if delays else np.nan
    f["avg_payment_delay"]          = float(np.clip(np.mean(delays), -10, 60)) if delays else np.nan
    f["missed_payments"]            = float(sum(1 for d in delays if d>7))
    f["payment_consistency_score"]  = float(np.clip((1-safe_cv([abs(d) for d in delays]))*100, 0, 100)) if delays else np.nan
    f["mobile_recharge_gap"]        = 20.0

    cursor.execute(
        "SELECT loan_amount, emi_amount, repayment_status FROM loans WHERE user_id=%s",
        (user_id,)
    )
    loan_rows = cursor.fetchall()
    on_time = sum(1 for r in loan_rows if r[2]=="on_time")
    total_l = len(loan_rows)
    f["informal_loans_count"]      = float(np.clip(total_l, 0, 10))
    f["informal_repayment_ratio"]  = float(on_time/total_l) if total_l else np.nan
    f["bnpl_usage_freq"]           = float(np.clip(total_l*0.5, 0, 10))
    f["bnpl_on_time_ratio"]        = float(on_time/total_l) if total_l else np.nan
    f["short_term_borrow_freq"]    = float(np.clip(total_l*0.3, 0, 6))
    f["repayment_borrow_ratio"]    = float(on_time/total_l) if total_l else np.nan
    f["emi_burden_ratio"]          = float(sum(r[1] for r in loan_rows if r[1])/(avg_inc*12)) if avg_inc and loan_rows else 0.2
    f["bounce_count"]              = float(sum(1 for r in loan_rows if r[2]=="missed"))
    f["loan_to_income_proxy"]      = float(sum(r[0] for r in loan_rows if r[0])/(avg_inc*12)) if avg_inc and loan_rows else 3.0

    f["avg_monthly_sales"]          = avg_inc
    f["sales_growth_rate"]          = f["income_trend"] if f["income_trend"] else 0.0
    f["order_cancellation_rate"]    = -0.3
    f["repeat_customer_ratio"]      = 0.4
    f["business_age_months"]        = 24.0

    cursor.execute(
        "SELECT age, employment_type, residence_type FROM users WHERE user_id=%s",
        (user_id,)
    )
    row = cursor.fetchone()
    if row:
        f["age"]               = float(row[0])
        f["residence_years"]   = float({"owned":12,"rented":3}.get(row[2], 5))
        f["employment_years"]  = float({"salaried":8,"business":6,"freelancer":4}.get(row[1], 5))
        f["salary_account_flag"] = float(1 if row[1]=="salaried" else 0)
    else:
        f["age"]               = np.nan
        f["residence_years"]   = np.nan
        f["employment_years"]  = np.nan
        f["salary_account_flag"] = np.nan

    f["upi_transaction_count"]       = float(np.clip(np.mean(txn)*0.08, 0, 38)) if txn else 10.0
    f["digital_txn_ratio"]           = float(np.clip(0.5 + (1-np.mean(sr) if sr else 0)*0.4, 0.1, 1.0))
    f["unique_merchants_per_month"]  = float(np.clip(np.mean(txn)*0.04, 1, 39)) if txn else 7.0
    f["avg_txn_size"]                = float(avg_spd/np.mean(txn)) if avg_spd and txn else 2000.0
    f["account_tenure_months"]       = 60.0
    f["num_bank_accounts"]           = 2.0
    f["savings_ratio"]               = float(np.clip(1-np.mean(sr), -0.05, 0.70)) if sr else 0.3
    f["self_transfer_ratio"]         = 0.09
    f["night_txn_ratio"]             = 0.09
    f["credit_enquiry_count"]        = 1.0
    f["insurance_payment_ratio"]     = float(np.clip(f["utility_payment_ratio"] if f["utility_payment_ratio"] else 0.7, 0.28, 1.0))
    f["rd_sip_flag"]                 = 0.0
    f["overdraft_usage_flag"]        = 0.0
    f["tax_filing_flag"]             = 0.5
    f["gst_registered_flag"]         = 0.2

    return f

def score_user(user_id: str) -> dict:
    conn   = get_connection()
    cursor = conn.cursor()

    gmm_bundle = load_gmm("models/gmm_model.pkl")
    with open("models/xgb_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model     = bundle["model"]
    feat_cols = bundle["feature_cols"]

    fd = fetch_user_features(user_id, cursor)
    conn.close()
    if not fd:
        return {"error": f"No data for {user_id}"}

    imp   = impute_features(fd, gmm_bundle)
    vec   = np.array([imp.get(c, 0) for c in feat_cols]).reshape(1,-1)
    score = int(np.clip(round(float(model.predict(vec)[0])), 300, 900))
    band, dec = assign_band(score)

    return {"user_id": user_id, "score": score, "band": band, "decision": dec}
="Arsh@1106", database="Barclays_Bank"
    )

def assign_band(score):
    if score >= 750: return "A", "Auto-Approve"
    if score >= 620: return "B", "Approve with lower limit"
    if score >= 500: return "C", "Require additional collateral / review"
    return "D", "Reject"

def fetch_user_features(user_id: str, cursor) -> dict:
    cursor.execute(
        "SELECT monthly_income, monthly_spending, total_transactions "
        "FROM user_monthly_features WHERE user_id=%s ORDER BY month_index",
        (user_id,)
    )
    rows = cursor.fetchall()
    if not rows:
        return {}

    inc = [r[0] for r in rows if r[0] is not None]
    spd = [r[1] for r in rows if r[1] is not None]
    txn = [r[2] for r in rows if r[2] is not None]

    def safe_cv(arr):
        return float(np.std(arr)/np.mean(arr)) if arr and np.mean(arr)!=0 else np.nan

    f = {}
    avg_inc = float(np.mean(inc)) if inc else np.nan
    avg_spd = float(np.mean(spd)) if spd else np.nan
    sr      = [s/i for s,i in zip(spd,inc) if i and i>0]

    f["avg_monthly_income"]        = avg_inc
    f["income_std_dev"]            = float(np.std(inc)) if inc else np.nan
    f["income_trend"]              = float((inc[-1]-inc[0])/inc[0]) if len(inc)>1 and inc[0] else np.nan
    f["income_sources_count"]      = 1.0
    f["salary_days_variance"]      = float(np.clip(np.var(inc)/1e8, 0, 20)) if inc else np.nan
    f["avg_monthly_spend"]         = avg_spd
    f["spend_to_income_ratio"]     = float(np.clip(np.mean(sr), 0.1, 2.0)) if sr else np.nan
    f["discretionary_spend_ratio"] = float(np.clip(np.mean(sr)*0.5, 0.04, 0.8)) if sr else np.nan
    f["balance_low_days"]          = float(sum(1 for r in sr if r>0.9))
    f["cash_withdrawal_ratio"]     = 0.33

    cursor.execute(
        "SELECT bill_amount, payment_date, due_date FROM utilities WHERE user_id=%s",
        (user_id,)
    )
    util_rows = cursor.fetchall()
    bills, delays = [], []
    for r in util_rows:
        if r[0]: bills.append(float(r[0]))
        if r[1] and r[2]:
            delays.append((pd.to_datetime(r[1])-pd.to_datetime(r[2])).days)

    f["utility_payment_ratio"]      = float(sum(1 for d in delays if d<=0)/len(delays)) if delays else np.nan
    f["avg_payment_delay"]          = float(np.clip(np.mean(delays), -10, 60)) if delays else np.nan
    f["missed_payments"]            = float(sum(1 for d in delays if d>7))
    f["payment_consistency_score"]  = float(np.clip((1-safe_cv([abs(d) for d in delays]))*100, 0, 100)) if delays else np.nan
    f["mobile_recharge_gap"]        = 20.0

    cursor.execute(
        "SELECT loan_amount, emi_amount, repayment_status FROM loans WHERE user_id=%s",
        (user_id,)
    )
    loan_rows = cursor.fetchall()
    on_time = sum(1 for r in loan_rows if r[2]=="on_time")
    total_l = len(loan_rows)
    f["informal_loans_count"]      = float(np.clip(total_l, 0, 10))
    f["informal_repayment_ratio"]  = float(on_time/total_l) if total_l else np.nan
    f["bnpl_usage_freq"]           = float(np.clip(total_l*0.5, 0, 10))
    f["bnpl_on_time_ratio"]        = float(on_time/total_l) if total_l else np.nan
    f["short_term_borrow_freq"]    = float(np.clip(total_l*0.3, 0, 6))
    f["repayment_borrow_ratio"]    = float(on_time/total_l) if total_l else np.nan
    f["emi_burden_ratio"]          = float(sum(r[1] for r in loan_rows if r[1])/(avg_inc*12)) if avg_inc and loan_rows else 0.2
    f["bounce_count"]              = float(sum(1 for r in loan_rows if r[2]=="missed"))
    f["loan_to_income_proxy"]      = float(sum(r[0] for r in loan_rows if r[0])/(avg_inc*12)) if avg_inc and loan_rows else 3.0

    f["avg_monthly_sales"]          = avg_inc
    f["sales_growth_rate"]          = f["income_trend"] if f["income_trend"] else 0.0
    f["order_cancellation_rate"]    = -0.3
    f["repeat_customer_ratio"]      = 0.4
    f["business_age_months"]        = 24.0

    cursor.execute(
        "SELECT age, employment_type, residence_type FROM users WHERE user_id=%s",
        (user_id,)
    )
    row = cursor.fetchone()
    if row:
        f["age"]               = float(row[0])
        f["residence_years"]   = float({"owned":12,"rented":3}.get(row[2], 5))
        f["employment_years"]  = float({"salaried":8,"business":6,"freelancer":4}.get(row[1], 5))
        f["salary_account_flag"] = float(1 if row[1]=="salaried" else 0)
    else:
        f["age"]               = np.nan
        f["residence_years"]   = np.nan
        f["employment_years"]  = np.nan
        f["salary_account_flag"] = np.nan

    f["upi_transaction_count"]       = float(np.clip(np.mean(txn)*0.08, 0, 38)) if txn else 10.0
    f["digital_txn_ratio"]           = float(np.clip(0.5 + (1-np.mean(sr) if sr else 0)*0.4, 0.1, 1.0))
    f["unique_merchants_per_month"]  = float(np.clip(np.mean(txn)*0.04, 1, 39)) if txn else 7.0
    f["avg_txn_size"]                = float(avg_spd/np.mean(txn)) if avg_spd and txn else 2000.0
    f["account_tenure_months"]       = 60.0
    f["num_bank_accounts"]           = 2.0
    f["savings_ratio"]               = float(np.clip(1-np.mean(sr), -0.05, 0.70)) if sr else 0.3
    f["self_transfer_ratio"]         = 0.09
    f["night_txn_ratio"]             = 0.09
    f["credit_enquiry_count"]        = 1.0
    f["insurance_payment_ratio"]     = float(np.clip(f["utility_payment_ratio"] if f["utility_payment_ratio"] else 0.7, 0.28, 1.0))
    f["rd_sip_flag"]                 = 0.0
    f["overdraft_usage_flag"]        = 0.0
    f["tax_filing_flag"]             = 0.5
    f["gst_registered_flag"]         = 0.2

    return f

def score_user(user_id: str) -> dict:
    conn   = get_connection()
    cursor = conn.cursor()

    gmm_bundle = load_gmm("models/gmm_model.pkl")
    with open("models/xgb_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model     = bundle["model"]
    feat_cols = bundle["feature_cols"]

    fd = fetch_user_features(user_id, cursor)
    conn.close()
    if not fd:
        return {"error": f"No data for {user_id}"}

    imp   = impute_features(fd, gmm_bundle)
    vec   = np.array([imp.get(c, 0) for c in feat_cols]).reshape(1,-1)
    score = int(np.clip(round(float(model.predict(vec)[0])), 300, 900))
    band, dec = assign_band(score)

    return {"user_id": user_id, "score": score, "band": band, "decision": dec}
