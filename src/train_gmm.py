
import pandas as pd
import numpy as np
import joblib

from sklearn.mixture import GaussianMixture
from .process_data import process_training_data


# Load processed data
df, user_features, X, X_scaled, scaler, feature_cols = process_training_data()


# =============================
# Find best number of clusters
# =============================

bic_scores = []
cluster_range = range(10, 21)

for k in cluster_range:

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        init_params="kmeans",
        n_init=10,
        max_iter=500,
        random_state=42
    )

    gmm.fit(X_scaled)

    bic = gmm.bic(X_scaled)

    bic_scores.append(bic)

    print(f"Clusters: {k}  |  BIC: {bic}")


best_k = cluster_range[np.argmin(bic_scores)]

print("\nBest number of clusters:", best_k)


# =============================
# Train final model
# =============================

gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    init_params="kmeans",
    n_init=10,
    max_iter=500,
    random_state=42
)

gmm.fit(X_scaled)


# Save models
joblib.dump(gmm, "models/gmm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")


# =============================
# Predict clusters
# =============================

clusters = gmm.predict(X_scaled)

probs = gmm.predict_proba(X_scaled)


user_cluster_df = pd.DataFrame({
    "user_id": user_features["user_id"],
    "cluster": clusters
})


# Save cluster probabilities

for i in range(best_k):
    user_cluster_df[f"cluster_prob_{i}"] = probs[:, i]


user_cluster_df.to_csv("models/user_clusters.csv", index=False)


# =============================
# Cluster statistics
# =============================

user_features["cluster"] = clusters


cluster_means = user_features.groupby("cluster").mean(numeric_only=True)

cluster_means.to_csv("models/cluster_means.csv")

print("Cluster means saved")


print("\nGMM clustering complete")
print("Total clusters:", best_k)

















# import os
# import joblib
# import pandas as pd
# from sklearn.mixture import GaussianMixture

# from process_data import process_training_data

# GMM_PATH = "models/gmm_model.pkl"
# CLUSTER_MEANS_PATH = "models/cluster_means.csv"
# CLUSTER_SUMMARY_PATH = "models/user_clusters.csv"


# def train_gmm_and_save(n_components: int = 4):
#     _, user_features, X, X_scaled, _, _ = process_training_data()

#     gmm = GaussianMixture(
#         n_components=n_components,
#         covariance_type="full",
#         random_state=42
#     )
#     gmm.fit(X_scaled)

#     cluster_labels = gmm.predict(X_scaled)
#     cluster_probs = gmm.predict_proba(X_scaled)

#     user_cluster_df = user_features[["user_id"]].copy()
#     user_cluster_df["cluster"] = cluster_labels

#     # Save cluster probabilities too
#     for i in range(n_components):
#         user_cluster_df[f"cluster_prob_{i}"] = cluster_probs[:, i]

#     X_with_cluster = X.copy()
#     X_with_cluster["cluster"] = cluster_labels

#     cluster_means = X_with_cluster.groupby("cluster").mean()

#     os.makedirs("models", exist_ok=True)
#     joblib.dump(gmm, GMM_PATH)
#     cluster_means.to_csv(CLUSTER_MEANS_PATH)
#     user_cluster_df.to_csv(CLUSTER_SUMMARY_PATH, index=False)

#     print("GMM trained successfully")
#     print("Cluster means saved to:", CLUSTER_MEANS_PATH)
#     print("User cluster summary saved to:", CLUSTER_SUMMARY_PATH)
#     print(user_cluster_df.head())


# if __name__ == "__main__":
#     train_gmm_and_save(n_components=4)




