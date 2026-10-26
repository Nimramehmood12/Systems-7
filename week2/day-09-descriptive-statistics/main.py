"""Day 09: descriptive statistics with manual and library comparisons."""

import statistics
from collections import Counter

import numpy as np


def mean_manual(values: list[float]) -> float:
    return sum(values) / len(values)


def median_manual(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def mode_manual(values: list[float]) -> list[float]:
    counts = Counter(values)
    max_count = max(counts.values())
    return [value for value, count in counts.items() if count == max_count]


def variance_manual(values: list[float]) -> float:
    avg = mean_manual(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def std_dev_manual(values: list[float]) -> float:
    return variance_manual(values) ** 0.5


def main() -> None:
    values = [10, 12, 12, 15, 18, 20, 20, 20, 25]

    print("=== Descriptive Statistics ===")
    print(f"Data: {values}\n")

    print("Manual calculations:")
    print(f"- Mean: {mean_manual(values):.4f}")
    print(f"- Median: {median_manual(values):.4f}")
    print(f"- Mode(s): {mode_manual(values)}")
    print(f"- Variance (population): {variance_manual(values):.4f}")
    print(f"- Std. deviation (population): {std_dev_manual(values):.4f}")

    print("\nLibrary calculations (statistics / NumPy):")
    print(f"- Mean: {statistics.mean(values):.4f} / {np.mean(values):.4f}")
    print(f"- Median: {statistics.median(values):.4f} / {np.median(values):.4f}")
    print(f"- Mode: {statistics.mode(values)}")
    print(f"- Variance (population): {statistics.pvariance(values):.4f} / {np.var(values):.4f}")
    print(f"- Std. deviation (population): {statistics.pstdev(values):.4f} / {np.std(values):.4f}")


if __name__ == "__main__":
    main()