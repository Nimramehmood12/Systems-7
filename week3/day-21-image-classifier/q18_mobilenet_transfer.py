import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255
base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights="imagenet")
base.trainable = False
inputs = tf.keras.Input((32, 32, 3))
x = tf.keras.layers.Resizing(96, 96)(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
transfer = tf.keras.Model(inputs, outputs)
transfer.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
transfer.fit(x_train[:5000], y_train[:5000], epochs=2, batch_size=64, verbose=0)

scratch = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation="softmax")])
scratch.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
scratch.fit(x_train[:5000], y_train[:5000], epochs=2, batch_size=64, verbose=0)
print("transfer accuracy:", transfer.evaluate(x_test[:1000], y_test[:1000], verbose=0)[1])
print("scratch accuracy:", scratch.evaluate(x_test[:1000], y_test[:1000], verbose=0)[1])
pred = transfer.predict(x_test[:9], verbose=0).argmax(axis=1)
fig, axes = plt.subplots(3, 3)
for axis, image, label in zip(axes.flat, x_test[:9], pred):
    axis.imshow(image); axis.set_title(f"predicted: {label}"); axis.axis("off")
plt.tight_layout(); plt.show()
