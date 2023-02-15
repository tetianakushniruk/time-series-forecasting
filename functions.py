import numpy as np


def weighted_sum(self, inputs, weights):
    if len(inputs) != len(weights):
        raise ValueError
    return np.sum(np.array(inputs) * np.array(weights))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    y = sigmoid_derivative(x)
    return y * (1 - y)
