"""Compare classification and regression evaluation techniques."""
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["malignant", "benign"]))

for name, estimator in [
    ("Logistic Regression", LogisticRegression(max_iter=3000)),
    ("Random Forest", RandomForestClassifier(n_estimators=150, random_state=42)),
]:
    scores = cross_val_score(estimator, X, y, cv=5, scoring="f1")
    print(f"{name}: mean F1={scores.mean():.3f}, std={scores.std():.3f}")

Xr, yr = load_diabetes(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(Xr, yr, test_size=0.2, random_state=42)
reg = LinearRegression().fit(train_x, train_y)
reg_pred = reg.predict(test_x)
mse = mean_squared_error(test_y, reg_pred)
print("Regression MAE:", round(mean_absolute_error(test_y, reg_pred), 3))
print("Regression MSE:", round(mse, 3))
print("Regression RMSE:", round(np.sqrt(mse), 3))
print("Regression R2:", round(r2_score(test_y, reg_pred), 3))
