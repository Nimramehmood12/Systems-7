"""Day 08: ML foundations with a small executable example."""

from collections import Counter


AI_ML_DL = {
    "Artificial Intelligence": "Building systems that can perform tasks requiring human-like intelligence.",
    "Machine Learning": "A subset of AI where models learn patterns from data.",
    "Deep Learning": "A subset of ML that uses multi-layer neural networks.",
}

LEARNING_TYPES = {
    "Supervised": "Model learns from labeled data (features + target label).",
    "Unsupervised": "Model finds patterns in unlabeled data.",
    "Reinforcement": "Agent learns actions from rewards and penalties.",
}



def main() -> None:
    """Print key terms and run a tiny supervised learning style demonstration."""
    print("=== AI, ML, and Deep Learning ===")
    for name, meaning in AI_ML_DL.items():
        print(f"- {name}: {meaning}")

    print("\n=== Learning Types ===")
    for name, meaning in LEARNING_TYPES.items():
        print(f"- {name}: {meaning}")

    # Tiny dataset: each sample is [hours_studied, practice_tests_done], label is Pass/Fail.
    samples = [
        {"features": [1, 0], "label": "Fail"},
        {"features": [2, 1], "label": "Fail"},
        {"features": [3, 1], "label": "Pass"},
        {"features": [4, 2], "label": "Pass"},
        {"features": [5, 3], "label": "Pass"},
    ]

    print("\n=== Dataset Terms in Practice ===")
    print(f"- Dataset size: {len(samples)} samples")
    print("- Features: hours_studied, practice_tests_done")
    print("- Label/Target: exam result (Pass/Fail)")

    labels = [row["label"] for row in samples]
    print(f"- Label distribution: {dict(Counter(labels))}")

    # Very simple rule-based predictor for demonstration (not a trained model yet).
    new_sample = [3, 2]
    predicted_label = "Pass" if sum(new_sample) >= 4 else "Fail"
    print(f"\nNew sample features={new_sample} -> predicted target: {predicted_label}")


if __name__ == "__main__":
    main()
