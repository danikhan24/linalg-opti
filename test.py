import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler    

from neural_net import Network, DenseLayer, ActivationLayer
from functions import mse, mse_grad, tanh, tanh_grad
from posixpath import split
from load_data import load_data
from dim_red import dim_reduction
from stratify import stratify_cv

def main():
    # Importing dataset
    X, y = load_data(split_xy=True)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = (
        stratify_cv(np.c_[dim_reduction(X, y, saveimg=False), y])
    )
    
    # Building the Network
    net = Network()
    net.add(DenseLayer(2, 10))
    net.add(ActivationLayer(tanh, tanh_grad))
    net.add(DenseLayer(10, 1))
    net.add(ActivationLayer(tanh, tanh_grad))

    # Fitting
    net.use(mse, mse_grad)
    net.fit(X_train, y_train, epochs=2000, learning_rate=0.001)

    # Testing
    results, err, acc = net.predict(X_test, y_test)


if __name__ == "__main__":
    main()

