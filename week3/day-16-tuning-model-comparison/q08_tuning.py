import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform

X, y = load_breast_cancer(return_X_y=True)
X_train, _, y_train, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
pipe = make_pipeline(StandardScaler(), SVC())
grid = GridSearchCV(pipe, {"svc__C": [.1, 1, 10], "svc__kernel": ["linear", "rbf"]}, cv=3, n_jobs=-1)
random = RandomizedSearchCV(pipe, {"svc__C": loguniform(.01, 100), "svc__gamma": loguniform(.0001, 1)},
                             n_iter=8, cv=3, random_state=42, n_jobs=-1)
for name, search in [("grid", grid), ("random", random)]:
    start = time.perf_counter(); search.fit(X_train, y_train)
    print(name, search.best_score_, search.best_params_, round(time.perf_counter()-start, 2), "seconds")
