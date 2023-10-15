import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def create_bar_graph(keys, values, label_x, label_y, title, name):
    plt.bar(keys, values, width=0.7)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.xticks(keys)
    for i in range(len(values)):
        plt.text(
            keys[i],
            values[i],
            str(round(values[i], 3)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.gcf().set_size_inches(1440 / 150, 720 / 150)
    plt.savefig("./graph_photos/{}".format(name), dpi=150)
    plt.clf()


def split_data():
    features = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "LABEL"]
    data_train = pd.read_csv(
        "./poker+hand/poker-hand-testing.data", delimiter=",", names=features
    )

    X = data_train.loc[:, data_train.columns != "LABEL"]
    Y = data_train["LABEL"]
    classes, classes_counts = np.unique(Y, return_counts=True)
    create_bar_graph(
        classes,
        classes_counts,
        "Classes",
        "Counts",
        "Class Distribution",
        "class_distribution.png",
    )

    return X, Y
