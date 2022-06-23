import pandas as pd
import numpy as np
from neural_net import Network, DenseLayer, ActivationLayer
from functions import mse, mse_grad, tanh, tanh_grad
from posixpath import split
from load_data import load_data
from dim_red import dim_reduction


def main():
    # Importing dataset
    X, y = load_data(split_xy=True)
    X = dim_reduction(X=X, y=y)
    print(X)

    x_train = X
    y_train = y

    # Building the Network

    net = Network()
    net.add(DenseLayer(2, 5))
    net.add(ActivationLayer(tanh, tanh_grad))
    net.add(DenseLayer(5, 1))
    net.add(ActivationLayer(tanh, tanh_grad))

    # Fitting
    net.use(mse, mse_grad)
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    # Testing
    results, err, acc = net.predict(x_train, y_train)


if __name__ == "__main__":
    main()

