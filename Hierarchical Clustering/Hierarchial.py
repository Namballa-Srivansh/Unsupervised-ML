from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()

X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# visualize

# sns.scatterplot(x = X[:, 0], y = X[:, 2])
# plt.show()

Z = linkage(X_scaled, method="ward")

# plot
# plt.figure(figsize=(12, 6))
# dendrogram(Z)
# plt.xlabel("samples")
# plt.ylabel("distance")
# plt.title("Dendrogram for hierarchial clustering")
# plt.show()

# Clustering

agg = AgglomerativeClustering(n_clusters=2)
labels = agg.fit_predict(X_scaled)

sns.scatterplot(x = X[:, 0], y = X[:, 2], c=labels)
plt.show()