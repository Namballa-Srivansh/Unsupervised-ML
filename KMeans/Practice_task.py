import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data
y = iris.target

# sns.scatterplot(x = X[:, 0], y = X[:, 2])
# plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional => Dimensionality reduction using PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

# Elbow Method

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

# sns.lineplot(x = range(1, 11), y = wcss, marker = 'o')
# plt.show()

kmeans = KMeans(n_clusters=3, random_state=10)
labels = kmeans.fit_predict(pca_data)

sns.scatterplot(x = pca_data[:, 0], y = pca_data[:, 1],  c = labels)
sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], marker="x", c="red")
plt.show()