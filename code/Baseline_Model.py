#!/usr/bin/env python
# coding: utf-8

# ## Baseline 1. OLS (log price)
# - Y：`log1p(price)`
# - drop：`id`, `neighbourhood_group`, `latitude`, `longitude`
# - One-hot：`room_type`, `neighbourhood`, `host_response_time`（drop_first）

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm

# Load dataset
DATA_PATH = "/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/Listing_Honululu.csv"
df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset shape: {df.shape}")
print(df[['room_type', 'neighbourhood', 'host_response_time']].dtypes)



# In[2]:


# Price distribution by buckets
price = df["price"].astype(float)
print("Basic stats:")
print(price.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

bins = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, np.inf]
price_bucket = pd.cut(price, bins=bins, right=False)
print("\nFrequency by price buckets (right-open intervals):")
print(price_bucket.value_counts().sort_index())



# In[3]:


# Baseline 1: OLS on log(price)
target = "price"
drop_cols = ["id", "neighbourhood_group", "latitude", "longitude"]
categorical_cols = ["room_type", "neighbourhood", "host_response_time"]

model_df = df.copy()
cols_to_drop = [c for c in drop_cols if c in model_df.columns]
X = model_df.drop(columns=cols_to_drop + [target])
y_price = model_df[target].astype(float)
y = np.log1p(y_price)

X_enc = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
X_enc = X_enc.astype(float)

# 80/20 shuffle split train vs test
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, shuffle=True, random_state=42
)

X_train_const = sm.add_constant(X_train, has_constant="add")
X_test_const = sm.add_constant(X_test, has_constant="add")

ols_model = sm.OLS(y_train, X_train_const).fit()

print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
print(f"Encoded feature count: {X_enc.shape[1]}")
print("Target: log(price)")



# In[5]:


# Evaluate (log space + original price space)

# Train predictions (log space)
train_pred_log = ols_model.predict(X_train_const)
train_mse_log = mean_squared_error(y_train, train_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
train_mae_log = mean_absolute_error(y_train, train_pred_log)
train_r2 = ols_model.rsquared
train_r2_adj = ols_model.rsquared_adj

# Test predictions (log space)
test_pred_log = ols_model.predict(X_test_const)
test_mse_log = mean_squared_error(y_test, test_pred_log)
test_rmse_log = np.sqrt(test_mse_log)
test_mae_log = mean_absolute_error(y_test, test_pred_log)
test_r2_log = r2_score(y_test, test_pred_log)

test_osr2_log = 1 - ((y_test - test_pred_log) ** 2).sum() / ((y_test - y_train.mean()) ** 2).sum()

# Back-transform to price space
train_pred_price = np.expm1(train_pred_log)
test_pred_price = np.expm1(test_pred_log)
train_price = np.expm1(y_train)
test_price = np.expm1(y_test)

train_mse_price = mean_squared_error(train_price, train_pred_price)
train_rmse_price = np.sqrt(train_mse_price)
train_mae_price = mean_absolute_error(train_price, train_pred_price)

test_mse_price = mean_squared_error(test_price, test_pred_price)
test_rmse_price = np.sqrt(test_mse_price)
test_mae_price = mean_absolute_error(test_price, test_pred_price)
test_r2_price = r2_score(test_price, test_pred_price)

test_osr2_price = 1 - ((test_price - test_pred_price) ** 2).sum() / ((test_price - train_price.mean()) ** 2).sum()

print("Log-space metrics:")
print(
    f"Train MSE: {train_mse_log:0.4f} | RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2:0.3f} | Adj R^2: {train_r2_adj:0.3f}"
)
print(
    f"Test  MSE: {test_mse_log:0.4f} | RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f} | OSR^2: {test_osr2_log:0.3f}"
)

print("\nOriginal-price-space metrics:")
print(
    f"Train MSE: {train_mse_price:0.2f} | RMSE: {train_rmse_price:0.2f} | MAE: {train_mae_price:0.2f}"
)
print(
    f"Test  MSE: {test_mse_price:0.2f} | RMSE: {test_rmse_price:0.2f} | MAE: {test_mae_price:0.2f} | R^2: {test_r2_price:0.3f} | OSR^2: {test_osr2_price:0.3f}"
)

# Top coefficients by absolute value (log-price model)
coef_table = (
    ols_model.params.rename("coef")
    .reset_index()
    .rename(columns={"index": "feature"})
    .assign(abs_coef=lambda d: d["coef"].abs())
    .sort_values("abs_coef", ascending=False)
)
print("\nTop 15 coefficients by |coef|:")
print(coef_table.head(15))

print("\nOLS summary (truncated):")
print(ols_model.summary())



# In[76]:


# Export OLS report and coefficients (log-price model)
report_path = "/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/ols_log_report.txt"
coef_path = "/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/ols_log_coefs.csv"
metrics_path = "/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/ols_log_metrics.json"

with open(report_path, "w") as f:
    f.write(ols_model.summary().as_text())

coef_table.to_csv(coef_path, index=False)

metrics = {
    # log-space
    "train_mse_log": float(train_mse_log),
    "train_rmse_log": float(train_rmse_log),
    "train_mae_log": float(train_mae_log),
    "train_r2_log": float(train_r2),
    "train_r2_adj_log": float(train_r2_adj),
    "test_mse_log": float(test_mse_log),
    "test_rmse_log": float(test_rmse_log),
    "test_mae_log": float(test_mae_log),
    "test_r2_log": float(test_r2_log),
    "test_osr2_log": float(test_osr2_log),
    # price-space (back-transformed)
    "train_mse_price": float(train_mse_price),
    "train_rmse_price": float(train_rmse_price),
    "train_mae_price": float(train_mae_price),
    "test_mse_price": float(test_mse_price),
    "test_rmse_price": float(test_rmse_price),
    "test_mae_price": float(test_mae_price),
    "test_r2_price": float(test_r2_price),
    "test_osr2_price": float(test_osr2_price),
    # data shape
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features_encoded": int(X_enc.shape[1]),
}
import json
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved summary to: {report_path}")
print(f"Saved coefficients to: {coef_path}")
print(f"Saved metrics to: {metrics_path}")



# ## Baseline 2. Ridge (log price)
# - Y：`log1p(price)`
# - drop：`id`, `neighbourhood_group`, `latitude`, `longitude`
# - One-hot：`room_type`, `neighbourhood`, `host_response_time`（drop_first）
# 
# 
# 

# In[6]:


# Ridge (fixed alpha)
ridge_drop_cols = ["id", "neighbourhood_group", "latitude", "longitude"]
ridge_cat_cols = ["room_type", "neighbourhood", "host_response_time"]

ridge_df = df.copy()
ridge_cols_to_drop = [c for c in ridge_drop_cols if c in ridge_df.columns]
X_ridge = ridge_df.drop(columns=ridge_cols_to_drop + ["price"])
y_price_ridge = ridge_df["price"].astype(float)
y_log_ridge = np.log1p(y_price_ridge)

X_ridge_enc = pd.get_dummies(X_ridge, columns=ridge_cat_cols, drop_first=True, dtype=float).astype(float)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_ridge_enc, y_log_ridge, test_size=0.2, shuffle=True, random_state=42
)

ridge_alpha = 10.0
ridge_model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=ridge_alpha))
ridge_model.fit(X_train_r, y_train_r)

y_train_pred_log = ridge_model.predict(X_train_r)
y_test_pred_log = ridge_model.predict(X_test_r)

ridge_train_mse_log = mean_squared_error(y_train_r, y_train_pred_log)
ridge_test_mse_log = mean_squared_error(y_test_r, y_test_pred_log)
ridge_train_rmse_log = np.sqrt(ridge_train_mse_log)
ridge_test_rmse_log = np.sqrt(ridge_test_mse_log)
ridge_train_mae_log = mean_absolute_error(y_train_r, y_train_pred_log)
ridge_test_mae_log = mean_absolute_error(y_test_r, y_test_pred_log)
ridge_train_r2_log = r2_score(y_train_r, y_train_pred_log)
ridge_test_r2_log = r2_score(y_test_r, y_test_pred_log)

print("Ridge (alpha=10) log-space metrics:")
print(
    f"Train RMSE: {ridge_train_rmse_log:0.4f} | MAE: {ridge_train_mae_log:0.4f} | R^2: {ridge_train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {ridge_test_rmse_log:0.4f} | MAE: {ridge_test_mae_log:0.4f} | R^2: {ridge_test_r2_log:0.3f}"
)

ridge_coefs = pd.DataFrame({
    "feature": X_ridge_enc.columns,
    "coef": ridge_model.named_steps["ridge"].coef_,
}).assign(abs_coef=lambda d: d.coef.abs()).sort_values("abs_coef", ascending=False)
print("\nTop 15 Ridge coefficients by |coef| (log-price space):")
print(ridge_coefs.head(15))



# ## Baseline 2a. Ridge CV (log price)
# - Same features as above，80/20 Train/Test split
# - Train set 5-fold CV Search alpha
# 
# 

# In[ ]:


# Ridge CV
ridge_df_cv = df.copy()
ridge_cols_to_drop_cv = [c for c in ["id", "neighbourhood_group", "latitude", "longitude"] if c in ridge_df_cv.columns]
X_ridge_cv = ridge_df_cv.drop(columns=ridge_cols_to_drop_cv + ["price"])
y_log_ridge_cv = np.log1p(ridge_df_cv["price"].astype(float))

X_ridge_enc_cv = pd.get_dummies(
    X_ridge_cv,
    columns=["room_type", "neighbourhood", "host_response_time"],
    drop_first=True,
    dtype=float,
).astype(float)

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
    X_ridge_enc_cv, y_log_ridge_cv, test_size=0.2, shuffle=True, random_state=42
)

alphas = np.logspace(-3, 3, 13)
ridge_cv_model = make_pipeline(StandardScaler(with_mean=False), Ridge())
param_grid = {"ridge__alpha": alphas}
search = GridSearchCV(ridge_cv_model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=1)
search.fit(X_train_cv, y_train_cv)

best_alpha = search.best_params_["ridge__alpha"]
best_model = search.best_estimator_
print(f"Best alpha: {best_alpha}")

y_train_pred_log = best_model.predict(X_train_cv)
y_test_pred_log = best_model.predict(X_test_cv)

train_mse_log = mean_squared_error(y_train_cv, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_cv, y_test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_cv, y_train_pred_log)
test_mae_log = mean_absolute_error(y_test_cv, y_test_pred_log)
train_r2_log = r2_score(y_train_cv, y_train_pred_log)
test_r2_log = r2_score(y_test_cv, y_test_pred_log)

print("Ridge CV log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)

ridge_cv_coefs = pd.DataFrame({
    "feature": X_ridge_enc_cv.columns,
    "coef": best_model.named_steps["ridge"].coef_,
}).assign(abs_coef=lambda d: d.coef.abs()).sort_values("abs_coef", ascending=False)
print("\nTop 15 Ridge-CV coefficients by |coef| (log-price space):")
print(ridge_cv_coefs.head(15))



# ## Baseline 3. Random Forest (log price)
# - Y：`log1p(price)`
# - drop：`id`, `neighbourhood_group`, `latitude`, `longitude`
# - One-hot：`room_type`, `neighbourhood`, `host_response_time`（drop_first）
# - 80/20 random split； GridSearchCV
# 
# 

# In[11]:


# Random Forest (fixed hyperparams)
rf_drop_cols = ["id", "neighbourhood_group", "latitude", "longitude"]
rf_cat_cols = ["room_type", "neighbourhood", "host_response_time"]

rf_df = df.copy()
rf_cols_to_drop = [c for c in rf_drop_cols if c in rf_df.columns]
X_rf = rf_df.drop(columns=rf_cols_to_drop + ["price"])
y_rf_log = np.log1p(rf_df["price"].astype(float))

X_rf_enc = pd.get_dummies(X_rf, columns=rf_cat_cols, drop_first=True, dtype=float).astype(float)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf_enc, y_rf_log, test_size=0.2, shuffle=True, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_rf, y_train_rf)

train_pred_log = rf_model.predict(X_train_rf)
test_pred_log = rf_model.predict(X_test_rf)

train_mse_log = mean_squared_error(y_train_rf, train_pred_log)
test_mse_log = mean_squared_error(y_test_rf, test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_rf, train_pred_log)
test_mae_log = mean_absolute_error(y_test_rf, test_pred_log)
train_r2_log = r2_score(y_train_rf, train_pred_log)
test_r2_log = r2_score(y_test_rf, test_pred_log)

print("RandomForest (fixed) log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)



# In[ ]:


# Random Forest GridSearchCV (log price)
param_grid = {
    "n_estimators": [50,100,200],
    "max_depth": [10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 8],
}

rf_base = RandomForestRegressor(
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)

rf_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    return_train_score=False,
)
rf_search.fit(X_train_rf, y_train_rf)

best_params = rf_search.best_params_
best_rf = rf_search.best_estimator_
print("Best params:", best_params)

y_train_pred_log = best_rf.predict(X_train_rf)
y_test_pred_log = best_rf.predict(X_test_rf)

train_mse_log = mean_squared_error(y_train_rf, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_rf, y_test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_rf, y_train_pred_log)
test_mae_log = mean_absolute_error(y_test_rf, y_test_pred_log)
train_r2_log = r2_score(y_train_rf, y_train_pred_log)
test_r2_log = r2_score(y_test_rf, y_test_pred_log)

print("RandomForest (GridSearchCV) log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)



# ## Baseline 4. LightGBM (log price) + GridSearchCV
# - Y：`log1p(price)`
# - Same features as above：drop `id`, `neighbourhood_group`, `latitude`, `longitude`；one-hot `room_type/neighbourhood/host_response_time`
# - 80/20 Train/Test；5-fold GridSearchCV
# 
# 

# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
# LightGBM Regressor with GridSearchCV (log price)
param_grid_lgbm = {
    "n_estimators": [200, 500, 800],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_samples": [10, 20, 40],
}

lgbm = LGBMRegressor(
    objective="regression",
    random_state=42,
    n_jobs=-1,
)

gs_lgbm = GridSearchCV(
    lgbm,
    param_grid_lgbm,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    return_train_score=False,
)
gs_lgbm.fit(X_train_rf, y_train_rf)

print("Best params (LightGBM):", gs_lgbm.best_params_)
best_lgbm = gs_lgbm.best_estimator_

y_train_pred_log = best_lgbm.predict(X_train_rf)
y_test_pred_log = best_lgbm.predict(X_test_rf)

train_mse_log = mean_squared_error(y_train_rf, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_rf, y_test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_rf, y_train_pred_log)
test_mae_log = mean_absolute_error(y_test_rf, y_test_pred_log)
train_r2_log = r2_score(y_train_rf, y_train_pred_log)
test_r2_log = r2_score(y_test_rf, y_test_pred_log)

print("LightGBM (GridSearchCV) log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)



# ## Random Forest (log price) — RandomizedSearchCV (faster, subsitute GridSearchCV)
# - Same features as above and same Y（80/20）
# - RandomizedSearchCV to save time
# 
# 

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Randomized search for RF (faster)
rand_param_dist = {
    "n_estimators": [50,100,200],
    "max_depth": [10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 8],
}

rf_base_rand = RandomForestRegressor(
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)

rf_rand = RandomizedSearchCV(
    rf_base_rand,
    rand_param_dist,
    n_iter=20,  
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    return_train_score=False,
    random_state=42,
)
rf_rand.fit(X_train_rf, y_train_rf)

best_params_rand = rf_rand.best_params_
best_rf_rand = rf_rand.best_estimator_
print("Best params (RandomizedSearchCV):", best_params_rand)

y_train_pred_log = best_rf_rand.predict(X_train_rf)
y_test_pred_log = best_rf_rand.predict(X_test_rf)

train_mse_log = mean_squared_error(y_train_rf, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_rf, y_test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_rf, y_train_pred_log)
test_mae_log = mean_absolute_error(y_test_rf, y_test_pred_log)
train_r2_log = r2_score(y_train_rf, y_train_pred_log)
test_r2_log = r2_score(y_test_rf, y_test_pred_log)

print("RandomForest (RandomizedSearchCV) log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)



# ## LightGBM (log price) — RandomizedSearchCV (faster)
# 
# 
# 

# In[ ]:


# LightGBM with RandomizedSearchCV (log price)
param_dist_lgbm = {
    "n_estimators": [200, 400, 800],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_samples": [10, 20, 40],
}

lgbm_base = LGBMRegressor(
    objective="regression",
    random_state=42,
    n_jobs=-1,
)

lgbm_rand = RandomizedSearchCV(
    lgbm_base,
    param_distributions=param_dist_lgbm,
    n_iter=25,  
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    return_train_score=False,
    random_state=42,
)
lgbm_rand.fit(X_train_rf, y_train_rf)

print("Best params (LGBM RandomizedSearchCV):", lgbm_rand.best_params_)
best_lgbm_rand = lgbm_rand.best_estimator_

y_train_pred_log = best_lgbm_rand.predict(X_train_rf)
y_test_pred_log = best_lgbm_rand.predict(X_test_rf)

train_mse_log = mean_squared_error(y_train_rf, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_rf, y_test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_rf, y_train_pred_log)
test_mae_log = mean_absolute_error(y_test_rf, y_test_pred_log)
train_r2_log = r2_score(y_train_rf, y_train_pred_log)
test_r2_log = r2_score(y_test_rf, y_test_pred_log)

print("LightGBM (RandomizedSearchCV) log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)



# ## Baseline 5. XGBoost (log price) — RandomizedSearchCV
# 
# 
# 

# In[ ]:


# XGBoost with RandomizedSearchCV (log price)
param_dist_xgb = {
    "n_estimators": [100, 200, 400, 800],
    "max_depth": [2, 3, 5, 7, 10],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
}

xgb_base = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

xgb_rand = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist_xgb,
    n_iter=25,  
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    return_train_score=False,
    random_state=42,
)

xgb_rand.fit(X_train_rf, y_train_rf)

print("Best params (XGBoost RandomizedSearchCV):", xgb_rand.best_params_)
best_xgb = xgb_rand.best_estimator_

y_train_pred_log = best_xgb.predict(X_train_rf)
y_test_pred_log = best_xgb.predict(X_test_rf)

train_mse_log = mean_squared_error(y_train_rf, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_rf, y_test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_rf, y_train_pred_log)
test_mae_log = mean_absolute_error(y_test_rf, y_test_pred_log)
train_r2_log = r2_score(y_train_rf, y_train_pred_log)
test_r2_log = r2_score(y_test_rf, y_test_pred_log)

print("XGBoost (RandomizedSearchCV) log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)



# ## Baseline 6. CatBoost (log price) — RandomizedSearchCV
# - 同样特征处理：丢弃 `id`, `neighbourhood_group`, `latitude`, `longitude`；one-hot `room_type/neighbourhood/host_response_time`
# - 目标：`log1p(price)`，80/20 划分
# - 随机搜索超参以加速

# In[ ]:


from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

# CatBoost with RandomizedSearchCV (log price)
cat_param_dist = {
    "n_estimators": [300, 500, 800, 1200],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bylevel": [0.7, 0.85, 1.0],
    "l2_leaf_reg": [1, 3, 5, 7, 10],
}

cat_base = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_state=42,
    verbose=False,
    thread_count=-1,
)

cat_rand = RandomizedSearchCV(
    estimator=cat_base,
    param_distributions=cat_param_dist,
    n_iter=25,  
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=1,  
    return_train_score=False,
    random_state=42,
)

cat_rand.fit(X_train_rf, y_train_rf)

print("Best params (CatBoost RandomizedSearchCV):", cat_rand.best_params_)
best_cat = cat_rand.best_estimator_

y_train_pred_log = best_cat.predict(X_train_rf)
y_test_pred_log = best_cat.predict(X_test_rf)

train_mse_log = mean_squared_error(y_train_rf, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_rf, y_test_pred_log)
train_rmse_log = np.sqrt(train_mse_log)
test_rmse_log = np.sqrt(test_mse_log)
train_mae_log = mean_absolute_error(y_train_rf, y_train_pred_log)
test_mae_log = mean_absolute_error(y_test_rf, y_test_pred_log)
train_r2_log = r2_score(y_train_rf, y_train_pred_log)
test_r2_log = r2_score(y_test_rf, y_test_pred_log)

print("CatBoost (RandomizedSearchCV) log-space metrics:")
print(
    f"Train RMSE: {train_rmse_log:0.4f} | MAE: {train_mae_log:0.4f} | R^2: {train_r2_log:0.3f}"
)
print(
    f"Test  RMSE: {test_rmse_log:0.4f} | MAE: {test_mae_log:0.4f} | R^2: {test_r2_log:0.3f}"
)


