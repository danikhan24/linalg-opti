from asyncio.constants import ACCEPT_RETRY_DELAY
import numpy as np
from sklearn.preprocessing import StandardScaler

from neural_nets.neural_net import Network, DenseLayer, ActivationLayer
from neural_nets.functions import mse, mse_grad, bce, bce_grad, relu, relu_grad, tanh, tanh_grad, sigmoid, sigmoid_grad
from neural_nets.decision_bound import plot_decision_boundary
from preprocessing.load_data import load_data
from preprocessing.dim_red import dim_reduction
from preprocessing.stratify import stratify_sampling


def define_network(width, depth, actf, actf_grad):
    # Defines a dense network with certain width and depth
    net = Network()
    net.add(DenseLayer(2, width))
    net.add(ActivationLayer(actf, actf_grad))
    for i in range(1, depth):
        net.add(DenseLayer(width, width))
        net.add(ActivationLayer(actf, actf_grad))
    net.add(DenseLayer(width, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_grad))
    return net

def main():
    # Importing dataset
    X, y = load_data(split_xy=True)

    # Normalize the first 13 columns
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Perform PCA and reduce its dimension to 2
    X_dim = dim_reduction(X, y, saveimg=True)

    # (Stratify-)split the data
    X_train, X_test, y_train, y_test = stratify_sampling(np.c_[X_dim, y])

    # Building the Network
    net = define_network(width=10, depth=3, actf=relu, actf_grad=relu_grad)

    # Fitting the Network
    net.use(bce, bce_grad)
    net.fit(X_train, y_train, epochs=2000, learning_rate=0.001)
    # Testing
    net.predict(X_test, y_test)

    # Plot the decision boundary of the 2-dim scatter plot
    plot_decision_boundary(X_dim, y, net, True, delta=0.01)


if __name__ == "__main__":
    main()
