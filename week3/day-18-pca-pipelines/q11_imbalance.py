from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1500, weights=[.95, .05], random_state=42)
a, b, c, d = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
for weights in [None, "balanced"]:
    model = LogisticRegression(class_weight=weights, max_iter=1000).fit(a, c)
    print(weights or "default", recall_score(d, model.predict(b)))
