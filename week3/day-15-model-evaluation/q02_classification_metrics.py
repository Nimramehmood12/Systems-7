import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
pred = LogisticRegression(max_iter=3000).fit(X_train, y_train).predict(X_test)
print("accuracy", accuracy_score(y_test, pred))
print("precision", precision_score(y_test, pred))
print("recall", recall_score(y_test, pred))
print("f1", f1_score(y_test, pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.tight_layout()
plt.show()
