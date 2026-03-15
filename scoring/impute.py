import pickle
import numpy as np

def load_gmm(path="models/gmm_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
import pickle
import numpy as np

def load_gmm(path="models/gmm_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def impute_features(partial_feats: dict, model_bundle: dict) -> dict:
    gmm    = model_bundle["gmm"]
    scaler = model_bundle["scaler"]
    pca    = model_bundle["pca"]
    cols   = model_bundle["feature_cols"]

    vec         = np.array([partial_feats.get(c, np.nan) for c in cols], dtype=float)
    missing_idx = np.where(np.isnan(vec))[0]
    avail_idx   = np.where(~np.isnan(vec))[0]

    if len(missing_idx) == 0:
        return {c: float(vec[i]) for i, c in enumerate(cols)}

    means      = scaler.mean_[avail_idx]
    stds       = scaler.scale_[avail_idx]
    vec_scaled = (vec[avail_idx] - means) / (stds + 1e-8)
    pca_load   = pca.components_[:, avail_idx]
    x_pca      = pca_load @ vec_scaled
    probs      = gmm.predict_proba(x_pca.reshape(1, -1))[0]

    cluster_means_scaled = pca.inverse_transform(gmm.means_)
    cluster_means_orig   = scaler.inverse_transform(cluster_means_scaled)

    filled = vec.copy()
    for idx in missing_idx:
        filled[idx] = float(np.dot(probs, cluster_means_orig[:, idx]))

    return {c: float(filled[i]) for i, c in enumerate(cols)}

def impute_features(partial_feats: dict, model_bundle: dict) -> dict:
    gmm    = model_bundle["gmm"]
    scaler = model_bundle["scaler"]
    pca    = model_bundle["pca"]
    cols   = model_bundle["feature_cols"]

    vec         = np.array([partial_feats.get(c, np.nan) for c in cols], dtype=float)
    missing_idx = np.where(np.isnan(vec))[0]
    avail_idx   = np.where(~np.isnan(vec))[0]

    if len(missing_idx) == 0:
        return {c: float(vec[i]) for i, c in enumerate(cols)}

    means      = scaler.mean_[avail_idx]
    stds       = scaler.scale_[avail_idx]
    vec_scaled = (vec[avail_idx] - means) / (stds + 1e-8)
    pca_load   = pca.components_[:, avail_idx]
    x_pca      = pca_load @ vec_scaled
    probs      = gmm.predict_proba(x_pca.reshape(1, -1))[0]

    cluster_means_scaled = pca.inverse_transform(gmm.means_)
    cluster_means_orig   = scaler.inverse_transform(cluster_means_scaled)

    filled = vec.copy()
    for idx in missing_idx:
        filled[idx] = float(np.dot(probs, cluster_means_orig[:, idx]))

    return {c: float(filled[i]) for i, c in enumerate(cols)}
