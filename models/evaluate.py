import sys, os, pickle
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputerimport sys, os, pickle
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

df = pd.read_csv("data/abh.csv")
drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment"]
df_feat      = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = [c for c in df_feat.columns if c != 'credit_score']
y = df_feat['credit_score'].values
X = df_feat[feature_cols].values

imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X)

with open("models/xgb_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model     = bundle["model"]
feat_cols = bundle["feature_cols"]

X_ord  = X_imp[:, [list(feature_cols).index(c) for c in feat_cols]]
y_pred = np.clip(model.predict(X_ord), 300, 900)

mae = mean_absolute_error(y, y_pred)
r2  = r2_score(y, y_pred)

def to_band(scores):
    return ["A" if s>=750 else "B" if s>=620 else "C" if s>=500 else "D"
            for s in scores]

ab  = to_band(y)
pb  = to_band(y_pred)
acc = sum(a==p for a,p in zip(ab,pb))/len(ab)*100

w25 = sum(1 for a,p in zip(y,y_pred) if abs(a-p)<=25)/len(y)*100
w50 = sum(1 for a,p in zip(y,y_pred) if abs(a-p)<=50)/len(y)*100

# AUC
y_bin  = (y >= 500).astype(int)
y_norm = (y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
_, Xt, _, yt = train_test_split(X_ord, y_bin,  test_size=0.2, random_state=42)
_, _,  _, yp = train_test_split(X_ord, y_norm, test_size=0.2, random_state=42)
auc = roc_auc_score(yt, yp)

print("=" * 50)
print("DARCRAYS — FINAL MODEL METRICS")
print("=" * 50)
print(f"AUC-ROC:       {auc:.4f}  ({'✓ ACHIEVED' if auc>=0.79 else '✗'}  target: 0.79+)")
print(f"MAE:           {mae:.1f} pts")
print(f"R² Score:      {r2:.4f}")
print(f"Band Accuracy: {acc:.1f}%")
print(f"Within ±25:    {w25:.1f}%")
print(f"Within ±50:    {w50:.1f}%")
print()
print(classification_report(ab, pb, target_names=["A","B","C","D"], zero_division=0))

from xgboost import XGBRegressor

df = pd.read_csv("data/abh.csv")
drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment"]
df_feat      = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = [c for c in df_feat.columns if c != 'credit_score']
y = df_feat['credit_score'].values
X = df_feat[feature_cols].values

imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X)

with open("models/xgb_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model     = bundle["model"]
feat_cols = bundle["feature_cols"]

X_ord  = X_imp[:, [list(feature_cols).index(c) for c in feat_cols]]
y_pred = np.clip(model.predict(X_ord), 300, 900)

mae = mean_absolute_error(y, y_pred)
r2  = r2_score(y, y_pred)

def to_band(scores):
    return ["A" if s>=750 else "B" if s>=620 else "C" if s>=500 else "D"
            for s in scores]

ab  = to_band(y)
pb  = to_band(y_pred)
acc = sum(a==p for a,p in zip(ab,pb))/len(ab)*100

w25 = sum(1 for a,p in zip(y,y_pred) if abs(a-p)<=25)/len(y)*100
w50 = sum(1 for a,p in zip(y,y_pred) if abs(a-p)<=50)/len(y)*100

# AUC
y_bin  = (y >= 500).astype(int)
y_norm = (y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
_, Xt, _, yt = train_test_split(X_ord, y_bin,  test_size=0.2, random_state=42)
_, _,  _, yp = train_test_split(X_ord, y_norm, test_size=0.2, random_state=42)
auc = roc_auc_score(yt, yp)

print("=" * 50)
print("DARCRAYS — FINAL MODEL METRICS")
print("=" * 50)
print(f"AUC-ROC:       {auc:.4f}  ({'✓ ACHIEVED' if auc>=0.79 else '✗'}  target: 0.79+)")
print(f"MAE:           {mae:.1f} pts")
print(f"R² Score:      {r2:.4f}")
print(f"Band Accuracy: {acc:.1f}%")
print(f"Within ±25:    {w25:.1f}%")
print(f"Within ±50:    {w50:.1f}%")
print()
print(classification_report(ab, pb, target_names=["A","B","C","D"], zero_division=0))
