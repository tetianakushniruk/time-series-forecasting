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

    def fit(self, X, y, max_epochs=1):
        self.X = X
        self.y = y
        self.all_outs.append(X)

        for epoch in range(0, max_epochs):
            print(self.__forward(X, y))


    def __forward(self, X, y):
        # outputs calculated with activation func on current layer
        # initial value - layer of inputs
        outputs = [X]
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num):
            # weighted sum (dot product of values and weights)
            S = np.dot(outputs[layer], self.neural_network.weights[layer])
            Y = sigmoid(S)
            outputs.append(Y)
        self.all_outs.append(outputs)
        return outputs

    def __back(self, Y, y, outputs):
        Y_ = sigmoid_derivative(Y)
        deltas = [np.array([[(Y - y) * Y_]])]
        for layer in range(self.neural_network.layers_num - 1, 0, -1):
            # calculate deltas for current layer
            delta = np.dot(deltas[-1], self.neural_network.weights[layer]) \
                    * sigmoid_derivative(np.array(outputs[layer]))
            deltas.append(delta.reshape(-1, 1))
        # reverse list of deltas because it was appended backwards
        deltas.reverse()
        return deltas

    def __update_weights(self, deltas):
        # iterate through dataset
        for i in range(0, len(self.all_outs)):
            # remove final output
            all_outs = np.array(self.all_outs[i][:-1])
            # iterate through layers of neural network
            for layer in range(0, self.neural_network.layers_num):

                print('weights before')
                print(self.neural_network.weights[layer])
                print(all_outs[layer])
                print(deltas[i][layer])
                print(all_outs[layer] * deltas[layer])
                self.neural_network.weights[layer] -= self.neural_network.learning_rate \
                                                      * np.array(self.all_outs[layer]) \
                                                      * deltas[layer]
                print('weights after')
                print(self.neural_network.weights[layer])

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
