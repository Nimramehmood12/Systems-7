import matplotlib.pyplot as plt
import tensorflow as tf

(images, _), _ = tf.keras.datasets.cifar10.load_data()
augment = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                               tf.keras.layers.RandomRotation(.1),
                               tf.keras.layers.RandomZoom(.1)])
sample = augment(images[:9] / 255.0)
fig, axes = plt.subplots(3, 3)
for axis, image in zip(axes.flat, sample):
    axis.imshow(image); axis.axis("off")
plt.tight_layout(); plt.show()
