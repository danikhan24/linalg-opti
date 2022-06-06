import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def dim_reduction(X, y, saveimg=True):
    pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.title("PCA applied to X")
    plt.scatter(pca.T[0][y == 1], pca.T[1][y == 1], label="heart disease")
    plt.scatter(
        pca.T[0][y == 0], pca.T[1][y == 0], color="red", label="no heart disease"
    )
    plt.legend()

    if saveimg:
        plt.savefig("figures/pca_n2.png", bbox_inches="tight")


def main():
    from load_data import load_data

    X, y = load_data(split_xy=True)
    dim_reduction(X, y, saveimg=False)


if __name__ == "__main__":
    main()
