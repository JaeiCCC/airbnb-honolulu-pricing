import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from scipy.spatial.distance import cdist


# Paths and feature definitions (mirrors baseline + clustering notebooks)
DATA_PATH = "/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/Listing_Honululu.csv"

cluster_features = [
    "latitude",
    "longitude",
    "drive_dist_hnl_km",
    "drive_dist_wk_km",
    "review_scores_rating",
    "review_scores_cleanliness",
    "accommodates",
    "bedrooms",
    "bathrooms",
]

# Baseline feature space (69 columns after one-hot with drop_first)
drop_cols = ["id", "neighbourhood_group", "latitude", "longitude"]
cat_cols = ["room_type", "neighbourhood", "host_response_time"]

# Target
TARGET = "price"

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape}")


def rmse(y_true, y_pred):
    # Older sklearn may not support squared=False
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def assign_to_centroid(X_scaled, centroids):
    """Assign each row in X_scaled to nearest centroid (Euclidean)."""
    distances = cdist(X_scaled, centroids, metric="euclidean")
    return distances.argmin(axis=1)


# Prepare baseline feature matrix (69 cols) and log-price target
model_df = df.copy()
cols_to_drop = [c for c in drop_cols if c in model_df.columns]
X_raw = model_df.drop(columns=cols_to_drop + [TARGET])
y_log = np.log1p(model_df[TARGET].astype(float))

X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float).astype(float)
print(f"Encoded feature count: {X_enc.shape[1]}")

# Train/test split must mirror baseline (80/20, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_log, test_size=0.2, shuffle=True, random_state=42
)
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")


# ---- Clustering (train-only fit) ----
clust_df = df[cluster_features].copy()
clust_train = clust_df.loc[X_train.index]
clust_test = clust_df.loc[X_test.index]

scaler = StandardScaler()
X_clust_train_scaled = scaler.fit_transform(clust_train)
X_clust_test_scaled = scaler.transform(clust_test)

# Silhouette check on train (k=2..9) to mirror notebook
sil_scores = {}
for k in range(2, 10):
    model_k = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_k = model_k.fit_predict(X_clust_train_scaled)
    sil_scores[k] = silhouette_score(X_clust_train_scaled, labels_k)

best_k = max(sil_scores, key=sil_scores.get)
print("Silhouette scores (train):", sil_scores)
print(f"Best k (train silhouette): {best_k}")

cluster_model = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
train_cluster_labels = cluster_model.fit_predict(X_clust_train_scaled)

# Compute centroids in scaled space and assign test by nearest centroid
centroids = np.vstack([
    X_clust_train_scaled[train_cluster_labels == c].mean(axis=0)
    for c in range(best_k)
])
test_cluster_labels = assign_to_centroid(X_clust_test_scaled, centroids)

print("Train cluster counts:", pd.Series(train_cluster_labels).value_counts().to_dict())
print("Test cluster counts:", pd.Series(test_cluster_labels).value_counts().to_dict())


# Hyperparameter search spaces (align with baseline RandomizedSearchCV choices)
rf_param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 8],
}

lgbm_param_dist = {
    "n_estimators": [200, 400, 800],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_samples": [10, 20, 40],
}


# Containers
cluster_metrics = []
rf_preds_test = pd.Series(index=X_test.index, dtype=float)
lgbm_preds_test = pd.Series(index=X_test.index, dtype=float)

for cluster_id in range(best_k):
    train_mask = pd.Series(train_cluster_labels, index=X_train.index) == cluster_id
    test_mask = pd.Series(test_cluster_labels, index=X_test.index) == cluster_id

    X_train_c = X_train.loc[train_mask]
    y_train_c = y_train.loc[train_mask]
    X_test_c = X_test.loc[test_mask]
    y_test_c = y_test.loc[test_mask]

    print(f"\nCluster {cluster_id}: train {len(X_train_c)}, test {len(X_test_c)}")

    # Random Forest
    rf_base = RandomForestRegressor(bootstrap=True, random_state=42, n_jobs=-1)
    rf_rand = RandomizedSearchCV(
        rf_base,
        rf_param_dist,
        n_iter=20,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        return_train_score=False,
    )
    rf_rand.fit(X_train_c, y_train_c)
    rf_train_pred = rf_rand.predict(X_train_c)
    rf_test_pred = rf_rand.predict(X_test_c) if len(X_test_c) else np.array([])

    rf_metrics = {
        "cluster": cluster_id,
        "model": "RF",
        "train_rmse": rmse(y_train_c, rf_train_pred),
        "train_mae": mae(y_train_c, rf_train_pred),
        "train_r2": r2(y_train_c, rf_train_pred),
        "test_rmse": rmse(y_test_c, rf_test_pred) if len(X_test_c) else np.nan,
        "test_mae": mae(y_test_c, rf_test_pred) if len(X_test_c) else np.nan,
        "test_r2": r2(y_test_c, rf_test_pred) if len(X_test_c) else np.nan,
    }
    cluster_metrics.append(rf_metrics)

    # LightGBM
    lgbm_base = LGBMRegressor(objective="regression", random_state=42, n_jobs=-1)
    lgbm_rand = RandomizedSearchCV(
        lgbm_base,
        param_distributions=lgbm_param_dist,
        n_iter=25,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        return_train_score=False,
    )
    lgbm_rand.fit(X_train_c, y_train_c)
    lgbm_train_pred = lgbm_rand.predict(X_train_c)
    lgbm_test_pred = lgbm_rand.predict(X_test_c) if len(X_test_c) else np.array([])

    lgbm_metrics = {
        "cluster": cluster_id,
        "model": "LGBM",
        "train_rmse": rmse(y_train_c, lgbm_train_pred),
        "train_mae": mae(y_train_c, lgbm_train_pred),
        "train_r2": r2(y_train_c, lgbm_train_pred),
        "test_rmse": rmse(y_test_c, lgbm_test_pred) if len(X_test_c) else np.nan,
        "test_mae": mae(y_test_c, lgbm_test_pred) if len(X_test_c) else np.nan,
        "test_r2": r2(y_test_c, lgbm_test_pred) if len(X_test_c) else np.nan,
    }
    cluster_metrics.append(lgbm_metrics)

    # Store test predictions for overall metrics
    rf_preds_test.loc[test_mask] = rf_test_pred
    lgbm_preds_test.loc[test_mask] = lgbm_test_pred

cluster_metrics_df = pd.DataFrame(cluster_metrics)
cluster_metrics_df


# ---- Overall metrics on test (log-price space) ----
rf_overall = {
    "model": "RF",
    "rmse": rmse(y_test, rf_preds_test),
    "mae": mae(y_test, rf_preds_test),
    "r2": r2(y_test, rf_preds_test),
}

lgbm_overall = {
    "model": "LGBM",
    "rmse": rmse(y_test, lgbm_preds_test),
    "mae": mae(y_test, lgbm_preds_test),
    "r2": r2(y_test, lgbm_preds_test),
}

overall_metrics_df = pd.DataFrame([rf_overall, lgbm_overall])
print("Cluster-level metrics (log-price):")
display(cluster_metrics_df)
print("\nOverall test metrics (log-price):")
display(overall_metrics_df)

