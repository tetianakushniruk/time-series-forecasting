import numpy as np
from numpy import sqrt
import pandas as pd

from NeuralNetwork import NeuralNetwork


class BackPropagation:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def fit(self, X, y, max_epochs=50000):
        y_pred = None
        errors = None
        E0 = 0
        final_epoch = max_epochs
        for epoch in range(0, max_epochs):
            # array of actual final outputs
            # shape is similar to array of expected outputs
            all_outs, y_pred = self.__forward(X)

            errors = (y_pred - y)**2
            E = np.sum(errors)
            if abs(E - E0) < 0.00001:
                final_epoch = epoch
                break
            E0 = E

            weight_deltas = self.__backward(y_pred, y, all_outs)
            self.__update_weights(weight_deltas)

        print(pd.DataFrame({"predicted": y_pred.ravel(), "actual": y.ravel(), 'error': errors.ravel()}))
        print('epoch: {}\ntotal error: {}'.format(final_epoch, np.sum(errors)))

    def __forward(self, X):
        # outputs calculated with activation func on current layer
        # initial value - layer of inputs
        all_outs = [X]
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num-1):
            S = np.dot(all_outs[layer], self.neural_network.weights[layer])
            Y = self.__activate(S)
            all_outs.append(Y)

        layer = self.neural_network.layers_num-1
        Y = np.dot(all_outs[layer], self.neural_network.weights[layer])
        return all_outs, Y

    def __backward(self, Y, y, all_outs):
        deltas = [np.array(Y - y)]

        for layer in range(self.neural_network.layers_num - 1, 0, -1):
            delta = np.dot(deltas[-1], self.neural_network.weights[layer].T) \
                    * self.__deactivate(all_outs[layer])
            deltas.append(delta)

        # reverse list of deltas because it was appended backwards
        deltas.reverse()
        for layer in range(0, self.neural_network.layers_num):
            deltas[layer] = self.neural_network.learning_rate \
                                     * np.dot(all_outs[layer].T, deltas[layer])
        return deltas

    def __update_weights(self, weight_deltas):
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num):
            self.neural_network.weights[layer] -= weight_deltas[layer]

    def test(self, X, y):
        a, y_pred = self.__forward(X)
        errors = (y_pred - y) ** 2
        print(pd.DataFrame({"predicted": y_pred.ravel(), "actual": y.ravel(), 'error': errors.ravel()}))
        print('total error: {}'.format(np.sum(errors)))
        return y_pred

    def predict(self, X):
        _, y_pred = self.__forward(X)
        print('prediction for {}: {:.2f}'.format(X, y_pred[0]))
        return y_pred[0]

    @staticmethod
    def __activate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __deactivate(x):
        return x * (1 - x)

    @property
    def neural_network(self):
        return self.__neural_network

    @neural_network.setter
    def neural_network(self, value):
        if not isinstance(value, NeuralNetwork):
            raise TypeError
        self.__neural_network = value
