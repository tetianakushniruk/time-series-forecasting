import numpy as np

class BackPropagation:
    def __init__(self, n=3):
        self.__X = None
        self.__y = None
        self.n = n

    def fit(self, X, y):
        self.X = X
        self.y = y
        pass

    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, value):
        if not isinstance(value, list):
            raise TypeError
        self.__validate_list(value)
        self.__X = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError
        self.__y = value

    def predict(self, X):
        self.X = X
        pass

    def __validate_list(self, list_):
        if not all(isinstance(x, (int, float)) for x in list_):
            raise TypeError
        if len(list_) != self.n:
            raise ValueError

    def __weighted_sum(self, inputs, weights):
        self.__validate_list(inputs)
        self.__validate_list(weights)
        return np.sum(np.array(inputs) * np.array(weights))

