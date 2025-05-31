import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

# Load dataset
def load_dataset():
    mat = pd.read_csv("codon_usage.csv")
    return mat.iloc[4:], mat.iloc[1]

# Load features and labels
X, y = load_dataset()
X = X.astype(float)  # Ensure numeric data

# Elbow Method to find optimal k
sse = []  # Sum of Squared Errors
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=2)
    km.fit(X)
    sse.append(km.inertia_)

# Plot SSE for different k
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, 11), y=sse)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors")
plt.title("Elbow Method for Optimal k")
plt.show()

# Apply KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(X)
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Predict cluster labels
pred = kmeans.predict(X)

# Plot clusters
plt.figure(figsize=(12, 5))

# First two features
plt.subplot(1, 2, 1)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=pred, cmap=cm.Accent)
for center in kmeans.cluster_centers_:
    plt.scatter(center[0], center[1], marker='^', c='red')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Cluster Plot (Feature 1 vs Feature 2)")
plt.grid(True)

# Next two features
plt.subplot(1, 2, 2)
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=pred, cmap=cm.Accent)
for center in kmeans.cluster_centers_:
    plt.scatter(center[2], center[3], marker='^', c='red')
plt.xlabel("Feature 3")
plt.ylabel("Feature 4")
plt.title("Cluster Plot (Feature 3 vs Feature 4)")
plt.grid(True)

plt.tight_layout()
plt.show()
