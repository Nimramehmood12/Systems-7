import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")])
model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
pred = model.predict(x_test, verbose=0).argmax(axis=1)
print("Test accuracy:", np.mean(pred == y_test.ravel()))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
