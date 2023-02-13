class BackPropagation:
    def __init__(self):
        self.X = None
        self.y = None

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
        if not all(isinstance(v, (int, float)) for v in value):
            raise TypeError
        if len(value) != 3:
            raise ValueError
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

