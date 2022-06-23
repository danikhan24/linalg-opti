import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def dim_reduction(X, y, saveimg=True):
    x = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit_transform(x)

    plt.figure(figsize=(8, 6))
    plt.title("PCA applied to X")
    plt.scatter(pca.T[0][y == 1], pca.T[1][y == 1], label="heart disease")
    plt.scatter(
        pca.T[0][y == 0], pca.T[1][y == 0], color="red", label="no heart disease"
    )
    plt.legend()

    if saveimg:
        plt.savefig("figures/pca_n2.png", bbox_inches="tight")

    principalDf = pd.DataFrame(
        data=pca, columns=["principal component 1", "principal component 2"]
    ).to_numpy()

    return principalDf
