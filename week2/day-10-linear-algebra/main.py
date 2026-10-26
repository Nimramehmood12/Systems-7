"""Day 10: linear algebra basics for ML."""

import numpy as np



def main() -> None:
    # Vectors can represent one sample with multiple features.
    vector_a = np.array([2, 4, 6])
    vector_b = np.array([1, 3, 5])

    # Matrices can represent many samples and many features.
    matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    dot_product = np.dot(vector_a, vector_b)

    # ML-style linear combination: prediction = w1*x1 + w2*x2 + w3*x3 + bias
    features = np.array([3.0, 2.0, 5.0])
    weights = np.array([0.8, 1.2, 1.5])
    bias = 2.0
    prediction = float(np.dot(features, weights) + bias)

    print("=== Vectors ===")
    print("vector_a:", vector_a)
    print("vector_b:", vector_b)

    print("\n=== Matrix ===")
    print(matrix)
    print("Matrix shape (rows, columns):", matrix.shape)

    print("\n=== Dot Product ===")
    print(f"vector_a · vector_b = {dot_product}")

    print("\n=== Linear Combination Example ===")
    print(f"features={features}, weights={weights}, bias={bias}")
    print(f"prediction = {prediction:.2f}")


if __name__ == "__main__":
    main()
