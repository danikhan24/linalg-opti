import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def dim_reduction(X, y, saveimg=True):
    print("Performing PCA on the data")
    pca = PCA(n_components=2).fit_transform(X).T
    plt.figure(figsize=(8, 6))
    plt.title("PCA applied to X")
    plt.scatter(pca[0][y == 0], pca[1][y == 0], label="no heart disease")    
    plt.scatter(pca[0][y == 1], pca[1][y == 1], color="red", label="heart disease")

    plt.legend()

    if saveimg:
        plt.savefig("figures/pca_n2.png", bbox_inches="tight")
        print("A plot of the transformed data is saved in the .../figures/ folder \n")

    return pca.T


def main():
    from load_data import load_data

    X, y = load_data(split_xy=True)
    print(dim_reduction(X, y, saveimg=False).shape)


if __name__ == "__main__":
    main()
