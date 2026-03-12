from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

import pandas as pd

from src.predict_credit import predict_credit_score_from_df

app = FastAPI(title="AI Alternate Credit Scoring API")


class MonthlyRecord(BaseModel):
    user_id: str
    year: int
    month: int
    age: Optional[int] = None
    employment_type: Optional[str] = None
    residence_type: Optional[str] = None
    city_tier: Optional[str] = None
    monthly_income: Optional[float] = None
    income_sources_count: Optional[int] = None
    income_std_6m: Optional[float] = None
    salary_day_variance: Optional[int] = None
    monthly_spending: Optional[float] = None
    spend_income_ratio: Optional[float] = None
    discretionary_spend_ratio: Optional[float] = None
    essential_spend_ratio: Optional[float] = None
    spending_volatility: Optional[float] = None
    balance_low_days: Optional[int] = None
    utility_bill_amount: Optional[float] = None
    utility_payment_ratio: Optional[float] = None
    utility_delay_days: Optional[int] = None
    bill_payment_consistency: Optional[float] = None
    bnpl_usage_count: Optional[int] = None
    bnpl_on_time_ratio: Optional[float] = None
    microloan_count: Optional[int] = None
    microloan_repayment_ratio: Optional[float] = None
    monthly_sales: Optional[float] = None
    sales_growth_rate: Optional[float] = None
    cashflow_variance: Optional[float] = None
    business_expense_ratio: Optional[float] = None
    late_payment_events: Optional[int] = None
    missed_payment_events: Optional[int] = None
    financial_stress_index: Optional[float] = None


@app.get("/")
def home():
    return {"message": "AI Alternate Credit Scoring API Running"}


@app.post("/predict")
def predict(records: List[MonthlyRecord]):
    df = pd.DataFrame([r.dict() for r in records])
    result = predict_credit_score_from_df(df)
    return result