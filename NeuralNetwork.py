import numpy as np


class NeuralNetwork:
    def __init__(self, nn_structure, learning_rate=0.1):
        self.nn_structure = nn_structure
        self.layers_num = len(nn_structure)
        self.learning_rate = learning_rate
        self.weights = [np.random.normal(scale=1, size=(self.nn_structure[i],
                                                        self.nn_structure[i + 1]))
                        for i in range(0, self.layers_num - 1)]
