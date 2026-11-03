from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
safe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
safe_scores = cross_val_score(safe, X_train, y_train, cv=5)
leaky_scaler = StandardScaler().fit(X)  # intentionally uses held-out rows
leaky_model = LogisticRegression(max_iter=3000)
leaky_scores = cross_val_score(leaky_model, leaky_scaler.transform(X_train), y_train, cv=5)
print("Leakage-safe CV:", safe_scores.mean(), "Leaky CV:", leaky_scores.mean())
