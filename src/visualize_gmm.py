import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from src.process_data import process_training_data

# Load processed data
df, user_features, X, X_scaled, scaler, feature_cols = process_training_data()

# Load cluster model
gmm = joblib.load("models/gmm_model.pkl")

# Load cluster assignments
cluster_df = pd.read_csv("models/user_clusters.csv")
clusters = cluster_df["cluster"]

print("Total users:", len(clusters))

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Get cluster centers
centers = gmm.means_
centers_pca = pca.transform(centers)

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="tab10",
    s=35,
    alpha=0.7
)

# Plot cluster centers
plt.scatter(
    centers_pca[:,0],
    centers_pca[:,1],
    c="black",
    s=200,
    marker="X",
    label="Cluster Centers"
)

plt.title("GMM Financial Behaviour Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.colorbar(scatter, label="Cluster")

plt.legend()
plt.grid(alpha=0.2)

plt.show()