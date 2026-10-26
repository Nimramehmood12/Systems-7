"""Day 12: simple and multiple linear regression with scikit-learn."""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



def evaluate_regression(model_name: str, y_true, y_pred) -> None:
    """Print common regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")



def main() -> None:
    print("Regression predicts continuous values (like prices).")
    print("Classification predicts categories (like pass/fail).\n")

    data = load_diabetes(as_frame=True)
    df = data.frame.copy()

    # Target for this dataset is disease progression (continuous value).
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simple linear regression using one feature.
    simple_feature = ["bmi"]
    simple_model = LinearRegression()
    simple_model.fit(X_train[simple_feature], y_train)
    simple_preds = simple_model.predict(X_test[simple_feature])

    print("=== Simple Linear Regression (feature: bmi) ===")
    print(f"Intercept: {simple_model.intercept_:.3f}")
    print(f"Coefficient for bmi: {simple_model.coef_[0]:.3f}")
    evaluate_regression("Simple model", y_test, simple_preds)

    # Multiple linear regression using all features.
    multiple_model = LinearRegression()
    multiple_model.fit(X_train, y_train)
    multiple_preds = multiple_model.predict(X_test)

    print("\n=== Multiple Linear Regression (all features) ===")
    print(f"Intercept: {multiple_model.intercept_:.3f}")
    coef_table = pd.Series(multiple_model.coef_, index=X.columns)
    print("Top coefficients by absolute value:")
    print(coef_table.abs().sort_values(ascending=False).head(5))
    evaluate_regression("Multiple model", y_test, multiple_preds)

    print("\nFirst 5 predictions from multiple model:")
    preview = pd.DataFrame({"actual": y_test.values[:5], "predicted": multiple_preds[:5]})
    print(preview.round(2))


if __name__ == "__main__":
    main()
