import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND
w = np.zeros(2); bias = 0.0
for _ in range(10):
    for row, target in zip(X, y):
        prediction = int(np.dot(row, w) + bias >= 0)
        error = target - prediction
        w += error * row
        bias += error
print("weights:", w, "bias:", bias)
print("predictions:", [int(np.dot(row, w) + bias >= 0) for row in X])
