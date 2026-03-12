import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from src.fetch_mysql_data import fetch_data
from src.feature_engineering_mysql import build_features


# -------------------------
# LOAD MODELS
# -------------------------

gmm = joblib.load("models/gmm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")


# -------------------------
# FETCH DATA
# -------------------------

users, transactions, loans, utilities, monthly = fetch_data()

features = build_features(users, transactions, loans, utilities, monthly)

X = features.drop(columns=["user_id"])
X = X[feature_cols]

X_scaled = scaler.transform(X)

clusters = gmm.predict(X_scaled)

features["cluster"] = clusters


# -------------------------
# PCA REDUCTION
# -------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# -------------------------
# SCATTER PLOT
# -------------------------

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="tab10",
    s=30,
    alpha=0.7
)

plt.title("Financial Behaviour Clusters (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.colorbar(scatter,label="Cluster ID")

plt.grid(alpha=0.3)

plt.show()