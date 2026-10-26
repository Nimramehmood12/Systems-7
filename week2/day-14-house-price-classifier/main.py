"""Day 14: house-price regression task + basic classification task."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def create_housing_dataset(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Create deterministic housing-style data for teaching tasks."""
    rng = np.random.default_rng(random_state)

    size_sqft = rng.integers(700, 3200, size=n_samples)
    bedrooms = rng.integers(1, 6, size=n_samples)
    age_years = rng.integers(0, 40, size=n_samples)
    distance_km = rng.uniform(1, 25, size=n_samples).round(2)

    # Synthetic price formula with controlled noise.
    noise = rng.normal(0, 12000, size=n_samples)
    price = (
        50000
        + size_sqft * 180
        + bedrooms * 12000
        - age_years * 1500
        - distance_km * 2200
        + noise
    )

    df = pd.DataFrame(
        {
            "size_sqft": size_sqft,
            "bedrooms": bedrooms,
            "age_years": age_years,
            "distance_km": distance_km,
            "price": price.round(2),
        }
    )
    return df


def run_regression_task(df: pd.DataFrame) -> None:
    """Task A: house-price prediction with simple and multiple regression."""
    print("=== Task A: House Price Regression ===")

    # Simple regression with one feature.
    X_simple = df[["size_sqft"]]
    y = df["price"]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    simple_model = LinearRegression()
    simple_model.fit(X_train_s, y_train_s)
    simple_preds = simple_model.predict(X_test_s)

    print("Simple Linear Regression")
    print(f"Intercept: {simple_model.intercept_:.2f}")
    print(f"Coefficient (size_sqft): {simple_model.coef_[0]:.2f}")
    print(f"MAE: {mean_absolute_error(y_test_s, simple_preds):.2f}")
    print(f"R2: {r2_score(y_test_s, simple_preds):.3f}")

    # Multiple regression with more features.
    X_multi = df[["size_sqft", "bedrooms", "age_years", "distance_km"]]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y, test_size=0.2, random_state=42
    )
    multi_model = LinearRegression()
    multi_model.fit(X_train_m, y_train_m)
    multi_preds = multi_model.predict(X_test_m)

    print("\nMultiple Linear Regression")
    print(f"Intercept: {multi_model.intercept_:.2f}")
    coef_table = pd.Series(multi_model.coef_, index=X_multi.columns)
    print("Coefficients:")
    print(coef_table.round(2))
    print(f"MAE: {mean_absolute_error(y_test_m, multi_preds):.2f}")
    print(f"R2: {r2_score(y_test_m, multi_preds):.3f}")

    preview = pd.DataFrame(
        {"actual_price": y_test_m.values[:5], "predicted_price": multi_preds[:5]}
    ).round(2)
    print("\nFirst 5 multiple-regression predictions:")
    print(preview)


def run_classification_task(df: pd.DataFrame) -> None:
    """Task B: basic classification with Random Forest."""
    print("\n=== Task B: Basic Classification Model ===")

    threshold = df["price"].median()
    df = df.copy()
    df["is_expensive"] = (df["price"] >= threshold).astype(int)

    X = df[["size_sqft", "bedrooms", "age_years", "distance_km"]]
    y = df["is_expensive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print("Classification report:")
    print(classification_report(y_test, preds))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("Feature importances:")
    print(importances.round(4))


def main() -> None:
    df = create_housing_dataset()
    run_regression_task(df)
    run_classification_task(df)


if __name__ == "__main__":
    main()