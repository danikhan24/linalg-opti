import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def relu(x):
    return (abs(x) + x) / 2


def relu_grad(x):
    return (x > 0) * 1


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    f = sigmoid(x)
    return f * (1 - f)
