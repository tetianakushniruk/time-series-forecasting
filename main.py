import numpy as np

from BackPropagation import BackPropagation
from NeuralNetwork import NeuralNetwork

series = [2.82, 3.48, 0.60, 4.76, 1.51, 5.51, 1.48, 5.19, 0.48, 5.22, 0.21, 4.19, 0.07, 4.63, 0.49]

inputs_num = 3
train_data_len = 13

X_train = []
y_train = []
for i in range(0, train_data_len-inputs_num):
    X_train.append(series[i:i+inputs_num])
    y_train.append(series[i+inputs_num])

X_train = np.array(X_train)
y_train = np.array([y_train]).reshape(-1, 1)

X_test = []
y_test = []
for i in range(train_data_len-inputs_num, len(series)-inputs_num):
    X_test.append(series[i:i+inputs_num])
    y_test.append(series[i+inputs_num])

X_test = np.array(X_test)
y_test = np.array([y_test]).reshape(-1, 1)

bp = BackPropagation(NeuralNetwork([3, 3, 1]))
print('TRAIN')
bp.fit(X_train, y_train)
print('TEST')
bp.test(X_test, y_test)

