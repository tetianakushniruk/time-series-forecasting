class BackPropagation:
    def __init__(self, data):
        self.data = data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        if not isinstance(value, list):
            raise TypeError
        if not len(value):
            raise ValueError
        self.__data = value

    def predict(self):
        pass


