from sklearn.metrics import silhouette_score
import joblib

from src.process_data import process_training_data

_, user_features, X, X_scaled, _, _ = process_training_data()

gmm = joblib.load("models/gmm_model.pkl")

labels = gmm.predict(X_scaled)

score = silhouette_score(X_scaled, labels)

print("Silhouette Score:", score)