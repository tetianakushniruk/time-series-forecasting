import numpy as np

from NeuralNetwork import NeuralNetwork
from functions import sigmoid, sigmoid_derivative


class BackPropagation:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.X = None
        self.y = None
        self.E = None
        self.all_outs = []
        self.all_deltas = []

    def fit(self, X, y, max_epochs=1):
        self.X = X
        self.y = y

        for epoch in range(0, max_epochs):
            self.__forward(X, y)

            # array of actual final outputs
            # shape is similar to array of expected outputs
            y_actual = self.all_outs[-1]

            self.__backward(y_actual, y)

            self.__update_weights()
            print(self.neural_network.weights)

    def __forward(self, X, y):
        # outputs calculated with activation func on current layer
        # initial value - layer of inputs
        self.all_outs.append(X)
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num):
            # weighted sum (dot product of values and weights)
            S = np.dot(self.all_outs[layer], self.neural_network.weights[layer])
            Y = sigmoid(S)
            self.all_outs.append(Y)

    def __backward(self, Y, y):
        Y_ = sigmoid_derivative(Y)
        self.all_deltas.append(np.array((Y - y) * Y_))

        for layer in range(self.neural_network.layers_num - 1, 0, -1):
            delta = np.dot(self.all_deltas[-1], self.neural_network.weights[layer].T) \
                    * sigmoid_derivative(self.all_outs[layer])
            self.all_deltas.append(delta)

        # reverse list of deltas because it was appended backwards
        self.all_deltas.reverse()

    def __update_weights(self):
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num):
            # print('weights before')
            # print(self.neural_network.weights[layer])

            self.neural_network.weights[layer] -= self.neural_network.learning_rate \
                                                  * np.dot(self.all_outs[layer].T,
                                                           self.all_deltas[layer])
            # print('weights after')
            # print(self.neural_network.weights[layer])

    def predict(self, X):
        self.X = X
        pass

    @property
    def neural_network(self):
        return self.__neural_network

    @neural_network.setter
    def neural_network(self, value):
        if not isinstance(value, NeuralNetwork):
            raise TypeError
        self.__neural_network = value
