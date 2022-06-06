import numpy as np
from load_data import load_data


def stratify_cv(data, n_splits=4):

    sorted_by_label = data[data[:, -1].argsort()]
    allocation = np.asarray(
        [sorted_by_label[i::n_splits] for i in range(n_splits)], dtype=object
    )
    training_data = np.vstack(allocation[: n_splits - 1])
    testing_data = allocation[n_splits - 1]

    np.random.shuffle(training_data)
    np.random.shuffle(testing_data)

    return (
        training_data[:, :-1],
        testing_data[:, :-1],
        training_data[:, -1],
        testing_data[:, -1],
    )


def main():
    _, _, y_train, y_test = stratify_cv(load_data(split_xy=False))
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))


if __name__ == "__main__":
    main()
