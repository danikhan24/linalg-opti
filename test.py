import pandas as pd
import numpy as np
from neural_net import Network, DenseLayer, ActivationLayer
from functions import mse, mse_grad, tanh, tanh_grad


def main():
    # Importing dataset

    df = pd.read_csv("heart.csv")
    print(df)
    df_x = df[
        [
            "0h",
            "1h",
            "2h",
            "3h",
            "4h",
            "5h",
            "6h",
            "7h",
            "8h",
            "9h",
            "10h",
            "11h",
            "12h",
        ]
    ]

    df_y = df[["13h"]]
    # THIS DOESN'T TRANSFORM DATA IN A SUITABLE FORMAT IDK WHAT TO DO
    x_train_1 = df_x.to_numpy()
    y_train_1 = df_y.to_numpy()

    # DATA HAVE TO BE LIKE THAT (btw, it is a xor problem)
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    # DATA HAVE TO BE LIKE THAT

    # Building the Network

    net = Network()
    net.add(DenseLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_grad))
    net.add(DenseLayer(3, 1))
    net.add(ActivationLayer(tanh, tanh_grad))

    # Fitting
    net.use(mse, mse_grad)
    net.fit(x_train, y_train, epochs=10000, learning_rate=0.001)

    # Testing
    out = net.predict(x_train)
    print(out)


if __name__ == "__main__":
    main()

