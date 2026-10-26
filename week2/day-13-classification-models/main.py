"""Day 13: compare common classification models."""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> None:
    """Fit a model and print accuracy and classification report."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.3f}")
    print("Classification report:")
    print(classification_report(y_test, predictions))


def main() -> None:
    data = load_wine(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test)

    # Feature importance is available for tree-based models.
    tree_model = models["Decision Tree"]
    forest_model = models["Random Forest"]

    tree_importance = pd.Series(tree_model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    forest_importance = pd.Series(
        forest_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 5 Decision Tree feature importances:")
    print(tree_importance.head(5).round(4))

    print("\nTop 5 Random Forest feature importances:")
    print(forest_importance.head(5).round(4))


if __name__ == "__main__":
    main()