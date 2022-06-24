import numpy as np
from preprocessing.load_data import load_data
from preprocessing.dim_red import dim_reduction


def stratify_sampling(data, n_splits=4):
    """ Stratified sampling
    Args:
        data:       the entire dataset (both X and y)
        n_split:    number of chunks the data is split into (n_splits=4 --> 75% / 25%)
    Returns:    Data split into training and testing set with equal representation of labels"""
    
    sorted_by_label = data[data[:, -1].argsort()]
    # allocate an equal amount of labels to n_splits chunks
    allocation = np.asarray(
        [sorted_by_label[i::n_splits] for i in range(n_splits)], dtype=object
    )
    # take all chunks except the last one as the training dataset
    training_data = np.vstack(allocation[: n_splits - 1])
    # final chunk is the testing dataset (with equal representation of labels)
    testing_data = allocation[n_splits - 1]

    # since the data gets sorted, remove the sorting by shuffling the data
    np.random.shuffle(training_data)
    np.random.shuffle(testing_data)

    return (
        training_data[:, :-1],
        testing_data[:, :-1],
        training_data[:, -1],
        testing_data[:, -1],
    )


def main():
    X, y = load_data(split_xy=True)
    X_train, X_test, y_train, y_test = stratify_sampling(
        np.c_[dim_reduction(X, y, saveimg=False), y]
    )

    print(X_train.shape)
    print(X_test.shape)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))


if __name__ == "__main__":
    main()
