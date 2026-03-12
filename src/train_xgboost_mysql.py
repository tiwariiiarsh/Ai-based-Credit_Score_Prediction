import xgboost as xgb
import joblib

def train_xgboost(df):

    X = df.drop(columns=["user_id","cluster"])

    # risk proxy
    y = df["spend_income_ratio"]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X,y)

    joblib.dump(model,"models/xgboost_model.pkl")

    print("XGBoost regression model trained")