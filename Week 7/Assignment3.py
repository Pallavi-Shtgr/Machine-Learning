import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Data Aggregation

# Sample data creation
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 70, 200),
    'tenure': np.random.randint(1, 10, 200),
    'monthly_spending': np.random.uniform(100, 1000, 200),
    'num_products': np.random.randint(1, 5, 200)
})

# Handling missing values (if any)
data.fillna(data.mean(), inplace=True)

# Normalize numerical features
scaler = StandardScaler()
data[['age', 'tenure', 'monthly_spending', 'num_products']] = scaler.fit_transform(data[['age', 'tenure', 'monthly_spending', 'num_products']])

# Plot distributions
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data['age'], kde=True, ax=axs[0, 0]).set(title='Age Distribution')
sns.histplot(data['tenure'], kde=True, ax=axs[0, 1]).set(title='Tenure Distribution')
sns.histplot(data['monthly_spending'], kde=True, ax=axs[1, 0]).set(title='Monthly Spending Distribution')
sns.histplot(data['num_products'], kde=True, ax=axs[1, 1]).set(title='Number of Products Distribution')
plt.tight_layout()
plt.show()

# Step 2: Clustering Using Hierarchical Clustering

# Select features for clustering
features = data[['age', 'tenure', 'monthly_spending', 'num_products']]

# Plot dendrogram to decide on the number of clusters
linked = linkage(features, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=10, show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Perform Agglomerative Clustering
n_clusters = 4  # Set based on dendrogram analysis
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
data['cluster'] = clustering.fit_predict(features)

# Step 3: Cluster Evaluation

# Summary statistics by cluster
cluster_summary = data.groupby('cluster').agg({
    'age': ['mean', 'median', 'std'],
    'tenure': ['mean', 'median', 'std'],
    'monthly_spending': ['mean', 'median', 'std'],
    'num_products': ['mean', 'median', 'std']
}).round(2)

print(cluster_summary)

# Step 4: Cluster Profiling

# Pairplot of clusters
sns.pairplot(data, hue='cluster', palette='viridis', diag_kind='kde')
plt.show()

# Scatter plot for selected features
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='age', y='monthly_spending', hue='cluster', palette='viridis')
plt.title('Clusters by Age and Monthly Spending')
plt.xlabel('Age')
plt.ylabel('Monthly Spending')
plt.show()
