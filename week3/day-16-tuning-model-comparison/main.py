"""Preprocess mixed data, tune models, and justify a final model."""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
models = {
    "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
    "svm": Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True))]),
    "forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "test F1:", round(f1_score(y_test, model.predict(X_test)), 3))

search = GridSearchCV(
    Pipeline([("scale", StandardScaler()), ("model", SVC())]),
    {"model__C": [.1, 1, 10], "model__kernel": ["linear", "rbf"], "model__gamma": ["scale", "auto"]},
    cv=5, scoring="f1", n_jobs=-1,
)
start = time.perf_counter()
search.fit(X_train, y_train)
print("Best SVM:", search.best_params_, round(search.best_score_, 3),
      "seconds:", round(time.perf_counter() - start, 2))
print("Chosen model test F1:", round(f1_score(y_test, search.predict(X_test)), 3))

# A small mixed-type example demonstrates the production preprocessing pattern.
frame = pd.DataFrame({"age": [20, 35, 44, 28], "city": ["A", "B", "A", "C"]})
prep = ColumnTransformer([("numeric", StandardScaler(), ["age"]),
                          ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"])])
print("Transformed mixed data shape:", prep.fit_transform(frame).shape)
