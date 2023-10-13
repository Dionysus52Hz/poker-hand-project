import pandas as pd
import numpy as np


def split_data():
    features = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "LABEL"]
    data_train = pd.read_csv(
        "./poker+hand/poker-hand-training-true.data", delimiter=",", names=features
    )

    X = data_train.loc[:, data_train.columns != "LABEL"]
    Y = data_train["LABEL"]

    test_class, test_counts = np.unique(Y, return_counts=True)
    print(test_class, test_counts)
    return X, Y
