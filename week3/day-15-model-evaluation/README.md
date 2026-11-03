# Day 1 - Model Evaluation

Use a stratified train-test split for classification and keep the test set
untouched until final evaluation. Accuracy is useful for balanced classes;
precision, recall, and F1 are better when the cost of false positives and false
negatives differs. A confusion matrix shows those error types directly, while
ROC-AUC summarizes ranking quality across classification thresholds.

For regression, MAE is easy to interpret, MSE/RMSE penalize large errors, and
R² measures explained variance. Cross-validation estimates how consistently a
model generalizes across different training subsets.

Run `python main.py`. Practice solutions are `q01_split.py` through `q05_roc.py`.
