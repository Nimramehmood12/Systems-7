import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=800, n_features=10, random_state=42)
X = StandardScaler().fit_transform(X)
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42)
def build(optimizer):
    model = tf.keras.Sequential([tf.keras.layers.Input((10,)), tf.keras.layers.Dense(24, activation="relu"),
                                 tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model
histories = {}
for name, optimizer in [("SGD", tf.keras.optimizers.SGD(.05)), ("Adam", tf.keras.optimizers.Adam(.001))]:
    histories[name] = build(optimizer).fit(a, c, validation_split=.2, epochs=15, verbose=0)
    print(name, histories[name].history["val_accuracy"][-1])
for name, history in histories.items():
    plt.plot(history.history["loss"], label=name)
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.show()
