import matplotlib.pyplot as plt
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
# COUNT USERS PER CLUSTER
# -------------------------

cluster_counts = features["cluster"].value_counts().sort_index()


# -------------------------
# BAR PLOT
# -------------------------

plt.figure(figsize=(10,6))

bars = plt.bar(cluster_counts.index, cluster_counts.values)


plt.title("Users per Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Users")


# write value on top of bars
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        int(height),
        ha='center',
        va='bottom'
    )


plt.tight_layout()

plt.show()