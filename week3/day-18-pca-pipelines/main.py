"""PCA and leakage-safe modeling patterns."""
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=.95)),
                 ("classifier", LogisticRegression(max_iter=3000))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
pipe.fit(X_train, y_train)
print("PCA components retaining 95% variance:", pipe.named_steps["pca"].n_components_)
print("Pipeline test accuracy:", pipe.score(X_test, y_test))

X_imb, y_imb = make_classification(n_samples=1500, weights=[.93, .07], random_state=42)
a, b, c, d = train_test_split(X_imb, y_imb, test_size=.25, stratify=y_imb, random_state=42)
for label, classifier in [("default", LogisticRegression(max_iter=1000)),
                          ("balanced", LogisticRegression(class_weight="balanced", max_iter=1000))]:
    classifier.fit(a, c)
    print(label, "minority recall:", round(recall_score(d, classifier.predict(b)), 3))
