import pickle, sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

CSV_PATH = "data/abh.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]} rows")

drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment"]
df_feat      = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = [c for c in df_feat.columns if c != 'credit_score']
y_all        = df_feat['credit_score'].values
X_all        = df_feat[feature_cols].values

# Resample to CIBIL 2024-25 distribution
TARGET = {
    (300,400): 0.04,
    (400,500): 0.12,
    (500,600): 0.39,
    (600,700): 0.20,
    (700,800): 0.16,
    (800,900): 0.09,
}
TOTAL = 15000

X_res, y_res = [], []
for (lo, hi), pct in TARGET.items():
    mask = (y_all >= lo) & (y_all < hi)
    Xs, ys = X_all[mask], y_all[mask]
    n = int(TOTAL * pct)
    idx = np.random.choice(len(Xs), size=n, replace=len(Xs)<n)
    X_res.append(Xs[idx])
    y_res.append(ys[idx])

X_final = np.vstack(X_res)
y_final = np.concatenate(y_res)
idx     = np.random.permutation(len(X_final))
X_final = X_final[idx]
y_final = y_final[idx]

print(f"\nTraining distribution (CIBIL matched, {len(X_final)} rows):")
bins = [300,400,500,600,700,800,900]
for i in range(len(bins)-1):
    c   = sum(1 for s in y_final if bins[i]<=s<bins[i+1])
    pct = c/len(y_final)*100
    bar = "█" * int(pct/2)
    print(f"  {bins[i]}-{bins[i+1]}: {c:5d} ({pct:.1f}%) {bar}")

# Impute
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X_final)

# Train GMM
print("\nTraining GMM...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
pca      = PCA(n_components=15)
X_pca    = pca.fit_transform(X_scaled)

best_gmm, best_bic = None, np.inf
for k in range(5, 12):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, max_iter=300)
    gmm.fit(X_pca)
    bic = gmm.bic(X_pca)
    print(f"  k={k}  BIC={bic:.1f}")
    if bic < best_bic:
        best_bic, best_gmm = bic, gmm

print(f"Best GMM: k={best_gmm.n_components}")
os.makedirs("models", exist_ok=True)
with open("models/gmm_model.pkl", "wb") as f:
    pickle.dump({
        "gmm": best_gmm, "scaler": scaler,
        "pca": pca, "feature_cols": feature_cols,
        "imputer": imputer
    }, f)
print("Saved → models/gmm_model.pkl")

# Train XGBoost
print("\nTraining XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_final, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2,
    objective="reg:squarederror", random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Test MAE: {mae:.1f} points")

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "model_type":   "regressor"
    }, f)
print("Saved → models/xgb_model.pkl")
import pickle, sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

CSV_PATH = "data/abh.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]} rows")

drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment"]
df_feat      = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = [c for c in df_feat.columns if c != 'credit_score']
y_all        = df_feat['credit_score'].values
X_all        = df_feat[feature_cols].values

# Resample to CIBIL 2024-25 distribution
TARGET = {
    (300,400): 0.04,
    (400,500): 0.12,
    (500,600): 0.39,
    (600,700): 0.20,
    (700,800): 0.16,
    (800,900): 0.09,
}
TOTAL = 15000

X_res, y_res = [], []
for (lo, hi), pct in TARGET.items():
    mask = (y_all >= lo) & (y_all < hi)
    Xs, ys = X_all[mask], y_all[mask]
    n = int(TOTAL * pct)
    idx = np.random.choice(len(Xs), size=n, replace=len(Xs)<n)
    X_res.append(Xs[idx])
    y_res.append(ys[idx])

X_final = np.vstack(X_res)
y_final = np.concatenate(y_res)
idx     = np.random.permutation(len(X_final))
X_final = X_final[idx]
y_final = y_final[idx]

print(f"\nTraining distribution (CIBIL matched, {len(X_final)} rows):")
bins = [300,400,500,600,700,800,900]
for i in range(len(bins)-1):
    c   = sum(1 for s in y_final if bins[i]<=s<bins[i+1])
    pct = c/len(y_final)*100
    bar = "█" * int(pct/2)
    print(f"  {bins[i]}-{bins[i+1]}: {c:5d} ({pct:.1f}%) {bar}")

# Impute
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X_final)

# Train GMM
print("\nTraining GMM...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
pca      = PCA(n_components=15)
X_pca    = pca.fit_transform(X_scaled)

best_gmm, best_bic = None, np.inf
for k in range(5, 12):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, max_iter=300)
    gmm.fit(X_pca)
    bic = gmm.bic(X_pca)
    print(f"  k={k}  BIC={bic:.1f}")
    if bic < best_bic:
        best_bic, best_gmm = bic, gmm

print(f"Best GMM: k={best_gmm.n_components}")
os.makedirs("models", exist_ok=True)
with open("models/gmm_model.pkl", "wb") as f:
    pickle.dump({
        "gmm": best_gmm, "scaler": scaler,
        "pca": pca, "feature_cols": feature_cols,
        "imputer": imputer
    }, f)
print("Saved → models/gmm_model.pkl")

# Train XGBoost
print("\nTraining XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_final, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2,
    objective="reg:squarederror", random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Test MAE: {mae:.1f} points")

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "model_type":   "regressor"
    }, f)
print("Saved → models/xgimport pickle, sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

CSV_PATH = "data/abh.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]} rows")

drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment"]
df_feat      = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = [c for c in df_feat.columns if c != 'credit_score']
y_all        = df_feat['credit_score'].values
X_all        = df_feat[feature_cols].values

# Resample to CIBIL 2024-25 distribution
TARGET = {
    (300,400): 0.04,
    (400,500): 0.12,
    (500,600): 0.39,
    (600,700): 0.20,
    (700,800): 0.16,
    (800,900): 0.09,
}
TOTAL = 15000

X_res, y_res = [], []
for (lo, hi), pct in TARGET.items():
    mask = (y_all >= lo) & (y_all < hi)
    Xs, ys = X_all[mask], y_all[mask]
    n = int(TOTAL * pct)
    idx = np.random.choice(len(Xs), size=n, replace=len(Xs)<n)
    X_res.append(Xs[idx])
    y_res.append(ys[idx])

X_final = np.vstack(X_res)
y_final = np.concatenate(y_res)
idx     = np.random.permutation(len(X_final))
X_final = X_final[idx]
y_final = y_final[idx]

print(f"\nTraining distribution (CIBIL matched, {len(X_final)} rows):")
bins = [300,400,500,600,700,800,900]
for i in range(len(bins)-1):
    c   = sum(1 for s in y_final if bins[i]<=s<bins[i+1])
    pct = c/len(y_final)*100
    bar = "█" * int(pct/2)
    print(f"  {bins[i]}-{bins[i+1]}: {c:5d} ({pct:.1f}%) {bar}")

# Impute
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X_final)

# Train GMM
print("\nTraining GMM...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
pca      = PCA(n_components=15)
X_pca    = pca.fit_transform(X_scaled)

best_gmm, best_bic = None, np.inf
for k in range(5, 12):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, max_iter=300)
    gmm.fit(X_pca)
    bic = gmm.bic(X_pca)
    print(f"  k={k}  BIC={bic:.1f}")
    if bic < best_bic:
        best_bic, best_gmm = bic, gmm

print(f"Best GMM: k={best_gmm.n_components}")
os.makedirs("models", exist_ok=True)
with open("models/gmm_model.pkl", "wb") as f:
    pickle.dump({
        "gmm": best_gmm, "scaler": scaler,
        "pca": pca, "feature_cols": feature_cols,
        "imputer": imputer
    }, f)
print("Saved → models/gmm_model.pkl")

# Train XGBoost
print("\nTraining XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_final, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2,
    objective="reg:squarederror", random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Test MAE: {mae:.1f} points")

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "model_type":   "regressor"
    }, f)
print("Saved → models/xgb_model.pkl")
import pickle, sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

CSV_PATH = "data/abh.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]} rows")

drop_cols = ["customer_id","profile_type","state_tier",
             "age_bucket","housing_type","risk_segment"]
df_feat      = df.drop(columns=[c for c in drop_cols if c in df.columns])
feature_cols = [c for c in df_feat.columns if c != 'credit_score']
y_all        = df_feat['credit_score'].values
X_all        = df_feat[feature_cols].values

# Resample to CIBIL 2024-25 distribution
TARGET = {
    (300,400): 0.04,
    (400,500): 0.12,
    (500,600): 0.39,
    (600,700): 0.20,
    (700,800): 0.16,
    (800,900): 0.09,
}
TOTAL = 15000

X_res, y_res = [], []
for (lo, hi), pct in TARGET.items():
    mask = (y_all >= lo) & (y_all < hi)
    Xs, ys = X_all[mask], y_all[mask]
    n = int(TOTAL * pct)
    idx = np.random.choice(len(Xs), size=n, replace=len(Xs)<n)
    X_res.append(Xs[idx])
    y_res.append(ys[idx])

X_final = np.vstack(X_res)
y_final = np.concatenate(y_res)
idx     = np.random.permutation(len(X_final))
X_final = X_final[idx]
y_final = y_final[idx]

print(f"\nTraining distribution (CIBIL matched, {len(X_final)} rows):")
bins = [300,400,500,600,700,800,900]
for i in range(len(bins)-1):
    c   = sum(1 for s in y_final if bins[i]<=s<bins[i+1])
    pct = c/len(y_final)*100
    bar = "█" * int(pct/2)
    print(f"  {bins[i]}-{bins[i+1]}: {c:5d} ({pct:.1f}%) {bar}")

# Impute
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X_final)

# Train GMM
print("\nTraining GMM...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
pca      = PCA(n_components=15)
X_pca    = pca.fit_transform(X_scaled)

best_gmm, best_bic = None, np.inf
for k in range(5, 12):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, max_iter=300)
    gmm.fit(X_pca)
    bic = gmm.bic(X_pca)
    print(f"  k={k}  BIC={bic:.1f}")
    if bic < best_bic:
        best_bic, best_gmm = bic, gmm

print(f"Best GMM: k={best_gmm.n_components}")
os.makedirs("models", exist_ok=True)
with open("models/gmm_model.pkl", "wb") as f:
    pickle.dump({
        "gmm": best_gmm, "scaler": scaler,
        "pca": pca, "feature_cols": feature_cols,
        "imputer": imputer
    }, f)
print("Saved → models/gmm_model.pkl")

# Train XGBoost
print("\nTraining XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_final, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2,
    objective="reg:squarederror", random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Test MAE: {mae:.1f} points")

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "model_type":   "regressor"
    }, f)
print("Saved → models/xgb_model.pkl")
b_model.pkl")
