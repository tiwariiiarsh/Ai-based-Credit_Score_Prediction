from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib

def train_gmm(df):

    X = df.drop(columns=["user_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 100 clusters
    gmm = GaussianMixture(
        n_components=40,
        covariance_type="full",
        random_state=42
    )

    clusters = gmm.fit_predict(X_scaled)

    df["cluster"] = clusters

    joblib.dump(gmm, "models/gmm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    return df