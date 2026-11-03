import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([200_000, 250_000, 300_000, 350_000])
y_pred = np.array([210_000, 245_000, 290_000, 360_000])
mae = np.mean(np.abs(y_true - y_pred))
mse = np.mean((y_true - y_pred) ** 2)
print("NumPy:", mae, mse, np.sqrt(mse), 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_true.mean())**2))
print("sklearn:", mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred),
      np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred))
