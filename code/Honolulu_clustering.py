#!/usr/bin/env python
# coding: utf-8

# ### Clustering

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv("/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/Listing_Honolulu.csv")
df.head()


# In[2]:


features_for_clustering = [
     "latitude",
     "longitude",
     "drive_dist_hnl_km",   # distance to HNL
     "drive_dist_wk_km",    # distance to Waikiki
     #"drive_time_hnl_min",
     #"drive_time_wk_min",
     "review_scores_rating",
     "review_scores_cleanliness",
    # "review_scores_location",
    # "reviews_per_month",
     "accommodates",
     "bedrooms",
     "bathrooms",
     #"host_listings_count"
    #"host_years"
    #"price"



    # "accommodates",
    # "beds",
    # "bedrooms",
    # "bathrooms",
    # "host_years",
    # "host_listings_count",
    # "calculated_host_listings_count",
    # "calculated_host_listings_count_entire_homes",
    # "host_response_rate",
    # "host_acceptance_rate",
    # "availability_365",
    # "availability_30",
    # "minimum_nights",
    # "maximum_nights",
    # "minimum_nights_avg_ntm",
    # "maximum_nights_avg_ntm",
    # "number_of_reviews",
    # "number_of_reviews_ltm",
    # "number_of_reviews_l30d",
    # "reviews_per_month",
    # "review_scores_rating",
    # "review_scores_cleanliness",
    # "review_scores_location",
    # "review_scores_communication",
    # "drive_dist_hnl_km",
    # "drive_time_hnl_min",
    # "drive_dist_wk_km",
    # "drive_time_wk_min",
    # "latitude",
    # "longitude",

]

clust_df = df[features_for_clustering].copy()
clust_df.describe()


# In[3]:


extra_features = [
    #"has_air_conditioning",
    #"has_gym",
    #"has_pool",

    #"is_room_in_hotel",
    #"is_entire_home"

    #"host_is_superhost"
]

features_for_clustering = features_for_clustering + extra_features
clust_df = df[features_for_clustering].copy()


# In[6]:


# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(clust_df)

X_scaled.shape


# In[7]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

sil_scores = {}

for k in range(2, 10):
    model_k = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"   # affinity/metric is fixed to euclidean for ward
    )
    labels_k = model_k.fit_predict(X_scaled)
    score_k = silhouette_score(X_scaled, labels_k)
    sil_scores[k] = score_k
    print(f"k = {k}, silhouette_score = {score_k:.3f}")



# In[8]:


import matplotlib.pyplot as plt

ks = list(sil_scores.keys())
scores = [sil_scores[k] for k in ks]

plt.plot(ks, scores, marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs k")
plt.show()


# In[9]:


# choose k based on the silhouette plot
best_k = 3

cluster_model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

cluster_labels = cluster_model.fit_predict(X_scaled)

# Attach to original df
df["Cluster_ID"] = cluster_labels
df["Cluster_ID"].value_counts()


# ### K-means robustness check

# In[10]:


clust_df.describe()
clust_df.nunique()


# In[11]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
km_labels = kmeans.fit_predict(X_scaled)

df["Cluster_ID_kmeans"] = km_labels
df[["Cluster_ID", "Cluster_ID_kmeans"]].head()


# In[12]:


clust_df.nunique()



# ## CLuster Profiling and Interpretation

# In[13]:


cluster_profile = df.groupby("Cluster_ID")[features_for_clustering + ["price"]].agg(
    ["mean", "median"]
)

cluster_profile


# 
# - Cluster 0 is the closest to waikiki
# - Cluster 1 has the highest average price

# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["Cluster_ID"],
    alpha=0.6
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Honolulu Listings by Cluster")
plt.colorbar(scatter, label="Cluster_ID")
plt.show()


# In[ ]:


import seaborn as sns

plt.figure(figsize=(16, 8))
sns.boxplot(
    data=df,
    x="Cluster_ID",
    y="price"
)
plt.yscale("log")  
plt.title("Price Distribution by Cluster")
plt.show()


# In[ ]:


output_cols = df.columns


df.to_csv("listing_honolulu_clustered.csv", index=False)


# ## PCA 
# 

# In[17]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


num_cols = [


    "latitude",
     "longitude",
     "drive_dist_hnl_km",   # distance to HNL
     "drive_dist_wk_km",    # distance to Waikiki
     #"drive_time_hnl_min",
     #"drive_time_wk_min",
     "review_scores_rating",
     "review_scores_cleanliness",
    # "review_scores_location",
    # "reviews_per_month",
     "accommodates",
     "bedrooms",
     "bathrooms",
]

pca_df = df[num_cols].copy()

scaler_pca = StandardScaler()
X_pca = scaler_pca.fit_transform(pca_df)

pca_full = PCA(random_state=42)
pca_full.fit(X_pca)

explained = pca_full.explained_variance_ratio_
cum_explained = explained.cumsum()

k_85 = int(np.argmax(cum_explained >= 0.85) + 1)
k_90 = int(np.argmax(cum_explained >= 0.90) + 1)
k_95 = int(np.argmax(cum_explained >= 0.95) + 1)

explained_df = pd.DataFrame(
    {
        "component": np.arange(1, len(explained) + 1),
        "explained_variance_ratio": explained,
        "cumulative_variance_ratio": cum_explained,
    }
)
explained_df.head()


# In[18]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))


axes[0].bar(explained_df["component"], explained_df["explained_variance_ratio"], color="#4c72b0")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Each Principal Component's Contribution")
axes[0].set_xticks(explained_df["component"])
axes[0].tick_params(axis="x", rotation=45)


axes[1].plot(explained_df["component"], explained_df["cumulative_variance_ratio"], marker="o")
axes[1].axhline(0.85, color="gray", linestyle="--", label="85%")
axes[1].axhline(0.90, color="red", linestyle="--", label="90%")
axes[1].axhline(0.95, color="green", linestyle="--", label="95%")
axes[1].axvline(k_85, color="gray", linestyle=":", alpha=0.7)
axes[1].axvline(k_90, color="red", linestyle=":", alpha=0.7)
axes[1].axvline(k_95, color="green", linestyle=":", alpha=0.7)
axes[1].annotate(f"k={k_85}", (k_85, cum_explained[k_85-1]), textcoords="offset points", xytext=(0,10), ha="center", color="gray")
axes[1].annotate(f"k={k_90}", (k_90, cum_explained[k_90-1]), textcoords="offset points", xytext=(0,10), ha="center", color="red")
axes[1].annotate(f"k={k_95}", (k_95, cum_explained[k_95-1]), textcoords="offset points", xytext=(0,10), ha="center", color="green")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Cumulated Contribution for first k principal components")
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"85% of the total variance: {k_85}")
print(f"90% of the total variance: {k_90}")
print(f"95% of the total variance: {k_95}")


# ## Use first 4 principles to do hierachical clustering
# 

# In[19]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


pc_scores = pca_full.transform(X_pca)
X_pca_13 = pc_scores[:, :4]

k_range = range(2, 9)
sil_records = []
for k in k_range:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_pca_13)
    sil = silhouette_score(X_pca_13, labels)
    sil_records.append({"k": k, "silhouette": sil})

sil_df = pd.DataFrame(sil_records)
sil_df


# In[20]:


plt.figure(figsize=(6, 4))
plt.plot(sil_df["k"], sil_df["silhouette"], marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Silhouette score")
plt.title("Silhouette vs k (Agglomerative on first 13 PCs)")
plt.grid(alpha=0.3)
plt.show()

best_k_13 = int(sil_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
print(f"Silhouette best k: {best_k_13}")

final_model_13 = AgglomerativeClustering(n_clusters=best_k_13, linkage="ward")
cluster_labels_13 = final_model_13.fit_predict(X_pca_13)

df["Cluster_PCA13"] = cluster_labels_13
df["Cluster_PCA13"].value_counts().sort_index()


# In[21]:


plt.figure(figsize=(6, 6))
scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["Cluster_PCA13"],
    alpha=0.6,
    cmap="tab10",
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Cluster distribution (k={best_k_13}, first 6 PCs)")
plt.colorbar(scatter, label="Cluster_PCA13")
plt.show()


# In[22]:


price_series = pd.to_numeric(df["price"], errors="coerce")

plt.figure(figsize=(10, 5))
sns.boxplot(data=df.assign(price_num=price_series), x="Cluster_PCA13", y="price_num")
plt.yscale("log")
plt.title("Price Distribution by Cluster (PCA13 Agglomerative)")
plt.xlabel("Cluster_PCA13")
plt.ylabel("Price (log scale)")
plt.show()



# In[ ]:




