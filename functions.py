import numpy as np


def weighted_sum(inputs, weights):
    if len(inputs) != len(weights):
        raise ValueError
    return np.dot(inputs, weights)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    y = sigmoid(x)
    return y * (1 - y)
