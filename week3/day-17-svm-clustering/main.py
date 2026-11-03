"""Classification and unsupervised clustering examples."""
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
for name, model in [("Naive Bayes", GaussianNB()), ("SVM", make_pipeline(StandardScaler(), SVC()))]:
    model.fit(X_train, y_train)
    print(name, "accuracy:", round(model.score(X_test, y_test), 3))

points, _ = make_blobs(n_samples=500, centers=4, cluster_std=.7, random_state=42)
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(points)
    print("k", k, "silhouette", round(silhouette_score(points, labels), 3))
db_labels = DBSCAN(eps=.7, min_samples=5).fit_predict(points)
print("DBSCAN clusters:", len(set(db_labels)) - (1 if -1 in db_labels else 0))
plt.scatter(points[:, 0], points[:, 1], c=db_labels, cmap="tab10")
plt.title("DBSCAN clusters (-1 is noise)")
plt.show()
