import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=.7, random_state=42)
inertia, silhouettes = [], []
for k in range(2, 11):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    inertia.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_)
    silhouettes.append(silhouette_score(X, labels))
print("Best silhouette k:", range(2, 11).__iter__().__next__() + max(range(len(silhouettes)), key=silhouettes.__getitem__))
fig, axes = plt.subplots(1, 2)
axes[0].plot(range(2, 11), inertia, marker="o"); axes[0].set_title("Elbow")
axes[1].plot(range(2, 11), silhouettes, marker="o"); axes[1].set_title("Silhouette")
plt.show()
