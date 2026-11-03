"""Multiclass image classification with a frozen pretrained base."""
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255
base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False,
                                         weights="imagenet")
base.trainable = False
inputs = tf.keras.Input((32, 32, 3))
x = tf.keras.layers.Resizing(96, 96)(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
model.fit(x_train[:10000], y_train[:10000], epochs=2, batch_size=64, validation_split=.1, verbose=1)
pred = model.predict(x_test[:2000], verbose=0).argmax(axis=1)
print(classification_report(y_test[:2000].ravel(), pred, zero_division=0))
