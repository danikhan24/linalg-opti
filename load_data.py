import pandas as pd


def load_data(split_xy=True):
    heart = pd.read_csv("heart.csv", header=None)
    # change the label '-1' to '0'
    heart.loc[heart[13] < 1, 13] = 0
    heart = heart.to_numpy()

    if split_xy:
        # split the data into X and y
        return heart[:, :-1], heart[:, -1]
    else:
        return heart
