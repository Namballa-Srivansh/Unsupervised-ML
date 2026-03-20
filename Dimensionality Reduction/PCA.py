import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()

X = pd.DataFrame(iris.data)
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# print("explained variance ratio: ", pca.explained_variance_ratio_)
# print(pca.components_)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA for iris dataset")

plt.show()