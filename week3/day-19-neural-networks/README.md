# Day 5 - Neural Networks

A perceptron computes a weighted sum plus bias and applies an activation.
Backpropagation uses the chain rule to send loss gradients backward, and
gradient descent updates weights. ReLU is a strong hidden-layer default,
sigmoid is useful for binary output, and softmax converts multiclass logits to
probabilities. Learning rate controls step size; batch size and epochs control
the update frequency and number of passes. Adam adapts step sizes, while SGD
can generalize well with suitable scheduling.

Run `python main.py`. Practice solutions are `q14_perceptron.py` and
`q15_keras_model.py`. TensorFlow is required for the Keras example.
