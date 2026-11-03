# Day 4 - PCA, Feature Engineering, Imbalance, and Leakage

The curse of dimensionality makes distance and density estimates less useful as
feature count grows. PCA rotates features into orthogonal components ordered by
explained variance, while feature selection removes irrelevant inputs. A
pipeline fits scaling, PCA, and the classifier only on each training fold,
preventing leakage from the test set. Accuracy can hide poor minority-class
performance, so recall and class weights matter for imbalanced data.

Run `python main.py`. Practice solutions are `q11_imbalance.py`,
`q12_pca.py`, and `q13_pipeline_leakage.py`.
