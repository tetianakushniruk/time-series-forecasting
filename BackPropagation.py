import numpy as np

from NeuralNetwork import NeuralNetwork


class BackPropagation:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.__X = None
        self.__y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        pass

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

    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, value):
        if not isinstance(value, list):
            raise TypeError
        validate_list(value)
        self.__X = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError
        self.__y = value


def validate_list(list_):
    if not all(isinstance(x, (int, float)) for x in list_):
        raise TypeError
