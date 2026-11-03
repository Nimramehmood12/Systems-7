import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
reduced = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
print("Explained variance:", PCA(n_components=2).fit(StandardScaler().fit_transform(X)).explained_variance_ratio_)
plt.scatter(reduced[:, 0], reduced[:, 1], c=y, cmap="viridis")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.show()
