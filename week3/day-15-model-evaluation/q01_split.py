from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Shapes:", X_train.shape, X_test.shape)
print("Train balance:", Counter(y_train))
print("Test balance:", Counter(y_test))
