"""Day 11: probability, correlation, and dataset preparation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    # Basic probability example.
    coin_outcomes = np.array(["H", "T", "H", "H", "T", "H", "T", "H", "H", "T"])
    probability_of_heads = np.mean(coin_outcomes == "H")
    print("=== Basic Probability ===")
    print(f"Estimated P(Heads) from 10 tosses: {probability_of_heads:.2f}")

    # Load local CSV from this folder.
    csv_path = Path(__file__).with_name("sample_data.csv")
    df = pd.read_csv(csv_path)

    print("\n=== Raw Dataset Preview ===")
    print(df.head())
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    # Convert invalid numeric values and clean missing values.
    df["exam_score"] = pd.to_numeric(df["exam_score"], errors="coerce")
    df["income"] = df["income"].fillna(df["income"].median())
    df["study_hours"] = df["study_hours"].fillna(df["study_hours"].median())
    df["exam_score"] = df["exam_score"].fillna(df["exam_score"].mean())

    print("\nMissing values after cleaning:")
    print(df.isna().sum())

    # Correlation for numeric columns.
    print("\n=== Correlation Matrix ===")
    numeric_df = df[["age", "income", "study_hours", "exam_score", "passed"]]
    print(numeric_df.corr().round(3))

    # Separate features and target.
    X = df[["age", "income", "study_hours", "exam_score"]]
    y = df["passed"]

    # 60/20/20 split with deterministic random state.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print("\n=== Split Sizes ===")
    print(f"Train: {len(X_train)} rows")
    print(f"Validation: {len(X_valid)} rows")
    print(f"Test: {len(X_test)} rows")


if __name__ == "__main__":
    main()