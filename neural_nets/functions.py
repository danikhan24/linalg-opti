import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def bce_grad(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return (1-y_true)/(1-y_pred) - y_true/y_pred


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