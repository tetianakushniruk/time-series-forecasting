import numpy as np

from NeuralNetwork import NeuralNetwork
from functions import weighted_sum, sigmoid


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
              print('---')
              print(X_train)
              self.__forward(X_train, y_train)

          print('epoch: {}\t'.format(epoch))

    def __forward(self, X, y):
        # outputs calculated with activation func on current layer
        outputs = []
        # iterate through layers of neural network
        for layer in range(0 , self.neural_network.layers_num):
            print('in layers')
            # outputs on current layer
            outputs_ = []
            # iterate through weight vectors of current layer
            for w in self.neural_network.weights[layer]:
                print('\tin weights')
                print(X)
                # calculate weighted sum (dot product)
                S = (weighted_sum(X, w))
                # activation func
                F = sigmoid(S)
                outputs_.append(F)
                print('\t\t'+str(S))
                print('\t\t'+str(F))
            X = outputs_
            outputs.append(outputs_)
        print(outputs)

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
