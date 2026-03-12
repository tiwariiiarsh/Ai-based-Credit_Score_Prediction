import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.process_data import process_training_data

# Load processed data
df, user_features, X, X_scaled, scaler, feature_cols = process_training_data()

# Load cluster assignments (1000 rows expected)
cluster_df = pd.read_csv("models/user_clusters.csv")
clusters = cluster_df["cluster"]

print("Total users:", len(clusters))

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="tab10",
    s=20,
    alpha=0.7
)

plt.title("GMM Financial Behaviour Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.colorbar(scatter, label="Cluster")

plt.grid(alpha=0.3)

plt.show()