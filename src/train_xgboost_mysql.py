import joblib
import xgboost as xgb

def train_model(df):

    X = df.drop(columns=["user_id", "cluster"])

    y = df["cluster"]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05
    )

    model.fit(X, y)

    joblib.dump(model, "models/xgboost_model.pkl")