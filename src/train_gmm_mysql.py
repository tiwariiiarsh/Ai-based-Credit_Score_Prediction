import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def train_gmm(df):

    X = df.drop(columns=["user_id"])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=4, random_state=42)

    clusters = gmm.fit_predict(X_scaled)

    df["cluster"] = clusters

    joblib.dump(gmm, "models/gmm_model.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

    return df