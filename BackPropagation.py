import numpy as np


class BackPropagation:
    def __init__(self, nn_structure, learning_rate=0.1):
        self.nn_structure = nn_structure
        self.layers_num = len(nn_structure)
        self.learning_rate = learning_rate
        self.weights = [np.random.normal(scale=1, size=(self.nn_structure[i],
                                                        self.nn_structure[i + 1]))
                        for i in range(0, self.layers_num - 1)]
        self.__X = None
        self.__y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        pass

    def predict(self, X):
        self.X = X
        pass

    def __validate_list(self, list_):
        if not all(isinstance(x, (int, float)) for x in list_):
            raise TypeError

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

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError
        self.__learning_rate = value
