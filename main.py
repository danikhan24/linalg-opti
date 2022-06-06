from posixpath import split
from load_data import load_data
from dim_red import dim_reduction


def main():
    X, y = load_data(split_xy=True)
    dim_reduction(X, y, saveimg=True)


if __name__ == "__main__":
    main()
