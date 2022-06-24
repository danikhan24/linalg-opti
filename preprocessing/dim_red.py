from posixpath import split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def dim_reduction(X, y, saveimg=True, img_name="pca_normalized"):
    """ Performs dimensionality reduction and plots resulting data as a scatter plot
    Args:
        X:          First 13 columns of hearts dataset
        y:          Target label (heart disease)
        saveimg:    saves image in .../figures/ folder when TRUE

    Returns:
        Transformed dataset represented by two features"""
    
    print("Performing PCA on the data")
    # reduce data to two columns
    pca = PCA(n_components=2).fit_transform(X).T
    plt.figure(figsize=(8, 6))
    plt.title("PCA applied to X")
    # scatter plot transformed data
    plt.scatter(pca[0][y == 0], pca[1][y == 0], label="no heart disease")
    plt.scatter(pca[0][y == 1], pca[1][y == 1], color="red", label="heart disease")

    plt.legend()

    if saveimg:
        # save scatter plot
        plt.savefig("figures/"+img_name+".png", bbox_inches="tight")
        print("A plot of the transformed data is saved in the .../figures/ folder \n")

    return pca.T


def main():
    from load_data import load_data

    X, y = load_data(split_xy=True)
    dim_reduction(X,y, saveimg=True, img_name="pca")


if __name__ == "__main__":
    main()
