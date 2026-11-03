"""Student-result prediction with a small Keras network."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit("Install week3 requirements to run this day.") from exc

X, y = make_classification(n_samples=1200, n_features=8, n_informative=5,
                           weights=[.45, .55], random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
model = tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],)),
                             tf.keras.layers.Dense(32, activation="relu"),
                             tf.keras.layers.Dropout(.2),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, validation_split=.2, epochs=20, batch_size=32, verbose=0)
print("Test loss and accuracy:", model.evaluate(X_test, y_test, verbose=0))
print("Best validation accuracy:", max(history.history["val_accuracy"]))
