# Week 2: Mathematics & Machine Learning Basics

This folder contains beginner-friendly daily practice for Week 2 of System 7.

## Daily Deliverables

- `day-08-ml-foundations/` - core ML concepts and a small executable dataset example
- `day-09-descriptive-statistics/` - mean, median, mode, variance, and standard deviation
- `day-10-linear-algebra/` - vectors, matrices, shape, and dot products
- `day-11-probability-data-prep/` - probability, correlation, data cleaning, and train/validation/test split
- `day-12-regression-sklearn/` - simple and multiple linear regression with scikit-learn
- `day-13-classification-models/` - Logistic Regression, KNN, Decision Tree, and Random Forest comparison
- `day-14-house-price-classifier/` - house-price regression task and a basic classification task

## Datasets Used

- Small in-script examples for foundational concepts
- Local CSV for Day 11 (`day-11-probability-data-prep/sample_data.csv`)
- Deterministic synthetic housing-style data generated in code for Day 14
- Built-in scikit-learn datasets for regression/classification days

## Dependencies

Install Week 2 dependencies from the repository root:

```bash
pip install -r week2/requirements.txt
```

## How Practical Tasks Map to Files

1. **Prepare dataset with Pandas/NumPy**
   - `day-11-probability-data-prep/main.py`
2. **Build house-price predictor**
   - `day-14-house-price-classifier/main.py` (regression section)
3. **Build classification model**
   - `day-13-classification-models/main.py`
   - `day-14-house-price-classifier/main.py` (classification section)

## Running All Day Scripts

From the repository root:

```bash
python week2/day-08-ml-foundations/main.py
python week2/day-09-descriptive-statistics/main.py
python week2/day-10-linear-algebra/main.py
python week2/day-11-probability-data-prep/main.py
python week2/day-12-regression-sklearn/main.py
python week2/day-13-classification-models/main.py
python week2/day-14-house-price-classifier/main.py
```