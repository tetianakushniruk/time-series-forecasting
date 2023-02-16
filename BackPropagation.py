import numpy as np

from NeuralNetwork import NeuralNetwork
from functions import weighted_sum, sigmoid, sigmoid_derivative


class BackPropagation:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.X = None
        self.y = None

    def fit(self, X, y, max_epochs=1):
        self.X = X
        self.y = y
        for epoch in range(0, max_epochs):
            for X_train, y_train in zip(X, y):
                result = self.__forward(X_train, y_train)
                y_output = result[-1][0]

                print('actual: {}\t\t\t\texpected: {}\t'.format(y_output, y_train))

                self.__back(Y=y_output, y=y_train, all_outs=result)
            print('epoch: {}\t'.format(epoch))

    def __forward(self, X, y):
        # outputs calculated with activation func on current layer
        outputs = []
        # iterate through layers of neural network
        for layer in range(0, self.neural_network.layers_num):
            # outputs on current layer
            outputs_ = []
            # iterate through weight vectors of current layer
            for w in self.neural_network.weights[layer]:
                # calculate weighted sum (dot product)
                S = (weighted_sum(X, w))
                # activation func
                Y = sigmoid(S)
                outputs_.append(Y)
            X = outputs_
            outputs.append(outputs_)
        return outputs

    def __back(self, Y, y, all_outs):
        delta = Y - y
        Y_ = sigmoid_derivative(Y)
        for layer in range(0, self.neural_network.layers_num-1):
            # calculate partial derivative of the error func for each output in layer
            # and store all results in list
            E_ = np.dot(delta * Y_, all_outs[layer])
            print('Weights before:')
            print(self.neural_network.weights[layer])
            for (w_i, w), e_ in zip(enumerate(self.neural_network.weights[layer]), E_):
                print(str(w_i) + ' ' + str(w) + ' ' + str(e_))
                self.neural_network.weights[layer][w_i] -= self.neural_network.learning_rate * e_
            print('Weights after:')
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
