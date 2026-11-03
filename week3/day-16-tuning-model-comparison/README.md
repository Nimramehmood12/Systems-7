# Day 2 - Tuning and Model Comparison

Underfitting has high bias; overfitting has high variance. Increasing tree
depth usually improves training performance but can reduce test performance.
Scaling numeric features and encoding categories must be learned only from the
training data, so `ColumnTransformer` belongs inside a pipeline. Grid search
checks an explicit parameter grid; randomized search samples combinations more
efficiently when the search space is large.

Run `python main.py`. Practice solutions are `q06_overfitting.py`,
`q07_preprocessing.py`, and `q08_tuning.py`.
