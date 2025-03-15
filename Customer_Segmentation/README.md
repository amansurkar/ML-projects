# Customer Segmentation using K-Means and DBSCAN

## Overview
This project applies **K-Means** and **DBSCAN** clustering algorithms to segment customers based on their behavior patterns. The dataset is preprocessed, standardized, and visualized using PCA to understand clustering patterns.

## Dataset
The dataset used for this project contains customer behavior data. It is assumed to have multiple numerical features representing customer attributes such as transaction amounts, frequency of purchases, etc.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Working of Clustering Algorithms

### K-Means Clustering
K-Means is a centroid-based clustering algorithm that works as follows:
1. Select the number of clusters (K).
2. Randomly initialize K cluster centroids.
3. Assign each data point to the nearest centroid.
4. Update centroids by computing the mean of assigned points.
5. Repeat steps 3-4 until convergence.

#### Advantages of K-Means:
- Works well for well-separated clusters.
- Fast and scalable.

#### Disadvantages of K-Means:
- Requires the number of clusters (K) to be predefined.
- Sensitive to outliers.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN is a density-based clustering algorithm that works as follows:
1. Define `eps` (neighborhood radius) and `min_samples` (minimum points to form a dense region).
2. Select a random unvisited point.
3. If it has at least `min_samples` points within `eps`, it forms a cluster.
4. Expand the cluster by adding neighboring points that meet the density criteria.
5. Repeat until all points are visited.
6. Points that do not belong to any cluster are classified as noise (-1).

#### Advantages of DBSCAN:
- Does not require the number of clusters to be specified.
- Handles clusters of arbitrary shape and noise.

#### Disadvantages of DBSCAN:
- Performance drops in high-dimensional data.
- Sensitive to `eps` and `min_samples` parameters.

## Implementation Steps
1. Load and preprocess the dataset.
2. Standardize the features using `StandardScaler`.
3. Apply K-Means clustering and assign labels.
4. Apply DBSCAN clustering and assign labels.
5. Use PCA for dimensionality reduction and visualization.
6. Save the segmented customer data to a CSV file.

## How to Run the Project
1. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Run the Python script:
   ```bash
   python customer_segmentation.py
   ```
3. The segmented dataset will be saved as `segmented_customers.csv`.

## Results
- Customers will be grouped into different clusters based on behavior patterns.
- The visualization will show how the customers are segmented using both K-Means and DBSCAN.

## Conclusion
- **K-Means** is useful for well-defined, spherical clusters.
- **DBSCAN** is effective for identifying noise and clusters of varying shapes.
- Both methods provide valuable insights into customer segmentation.


