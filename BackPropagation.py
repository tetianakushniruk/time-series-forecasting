import numpy as np

from NeuralNetwork import NeuralNetwork
from functions import weighted_sum, sigmoid, sigmoid_derivative


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
        for epoch in range(0, max_epochs):
            deltas = []
            for X_train, y_train in zip(X, y):
                self.__forward(X_train, y_train)
                y_output = self.all_outs[-1][0]

                print('actual: {}\t\t\t\texpected: {}\t'.format(y_output, y_train))

                # deltas.append(self.__back(Y=y_output, y=y_train))

                # self.__update_weights(deltas)
            # print(deltas)
            print('epoch: {}\t'.format(epoch))

    def __forward(self, X, y):
        # outputs calculated with activation func on current layer
        # initial value - layer of inputs
        outputs = [X]
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num):
            S = weighted_sum(outputs[layer], self.neural_network.weights[layer].T)
            Y = sigmoid(S)
            outputs.append(Y)
        self.all_outs = outputs
        print(outputs)

    def __back(self, Y, y):
        Y_ = sigmoid_derivative(Y)
        deltas = [[(Y - y) * Y_]]
        for layer in range(self.neural_network.layers_num - 1, 0, -1):
            # calculate deltas for current layer
            delta = np.dot(deltas[-1], self.neural_network.weights[layer]) \
                    * sigmoid_derivative(np.array(self.all_outs[layer]))
            deltas.append(delta.reshape(-1, 1))
        # reverse list of deltas because it was appended backwards
        deltas.reverse()
        return deltas

    def __update_weights(self, deltas):
        # remove the last output
        all_outs = self.all_outs[:-1]
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num):
            # print('weights before')
            # print(self.neural_network.weights[layer])
            print(np.array(all_outs))
            print(deltas)
            print(np.array(all_outs, deltas))
            self.neural_network.weights[layer] -= self.neural_network.learning_rate \
                                                  * np.array(self.all_outs[layer]) \
                                                  * deltas[layer]
            # print('weights after')
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
