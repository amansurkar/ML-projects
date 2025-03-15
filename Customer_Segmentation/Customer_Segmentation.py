import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Load dataset (example: customer behavior dataset)
df = pd.read_csv("customer_data.csv")

# Data Preprocessing
# Remove any missing values
df.dropna(inplace=True)

# Standardize the data to have a mean of 0 and variance of 1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 1:])  # Assuming first column is ID

# Applying K-Means Clustering
# K-Means is a centroid-based clustering method, setting k=4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# Applying DBSCAN Clustering
# DBSCAN is a density-based clustering algorithm
# eps defines the neighborhood radius, and min_samples defines the minimum points to form a cluster
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Visualizing Clusters using PCA
# Reduce data dimensions to 2 for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df['PCA1'], df['PCA2'] = df_pca[:, 0], df_pca[:, 1]

# Plot K-Means Clustering results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['KMeans_Cluster'], palette='viridis')
plt.title("K-Means Clustering")

# Plot DBSCAN Clustering results
plt.subplot(1, 2, 2)
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['DBSCAN_Cluster'], palette='viridis')
plt.title("DBSCAN Clustering")
plt.show()

# Save segmented data to a CSV file
df.to_csv("segmented_customers.csv", index=False)
print("Customer segmentation completed and saved!")
