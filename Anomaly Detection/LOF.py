import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

df = pd.read_csv("thyroid_dataset.csv")

X = df.drop("Outlier_label", axis=1)
y = df["Outlier_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

neighbs = LocalOutlierFactor(contamination=0.036)

labels = neighbs.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8, 6))

# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
# plt.xlabel("PC1")
# plt.ylabel("PC2")

# plt.show()

n_outliers = np.sum(labels == -1)
n_normal = np.sum(labels == 1)

print("outlier = ", n_outliers)
print("normal = ", n_normal)