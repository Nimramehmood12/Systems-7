from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)
models = [LogisticRegression(max_iter=3000), KNeighborsClassifier(), RandomForestClassifier(random_state=42)]
for model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(type(model).__name__, f"mean={scores.mean():.3f}", f"std={scores.std():.3f}")
