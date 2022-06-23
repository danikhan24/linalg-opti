import numpy as np
import matplotlib.pyplot as plt

from dim_red import dim_reduction


def decision_boundary(X, y, net, saveimg=True):
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1,r2))

    y_hat = np.round(net.forward_propagation(grid))

    zz = y_hat.reshape(xx.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, cmap='bwr', alpha=0.4)

    plt.title("Decision bound")
    plt.scatter(X.T[0][y == 0], X.T[1][y == 0], label="no heart disease")
    plt.scatter(X.T[0][y == 1], X.T[1][y == 1], color="red", label="heart disease")
    plt.legend()

    if saveimg:
        plt.savefig("figures/decision_bound.png", bbox_inches="tight")
        print("A plot of the decision bound is saved in the .../figures/ folder \n")
