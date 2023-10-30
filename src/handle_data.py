import pandas as pd
import numpy as np
import dataframe_image as dfi
from matplotlib import pyplot as plt


# Tao bieu do cot
def create_single_bar_plot(keys, values, label_x, label_y, title, saved_name):
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
    plt.savefig("image_export/{}".format(saved_name), dpi=150)
    plt.clf()


# Chia dataset ra 2 phan: thuoc tinh va nhan
def split_data():
    features = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "LABEL"]
    data_train = pd.read_csv(
        "dataset/poker-hand-testing.data", delimiter=",", names=features
    )

    X = data_train.loc[:, data_train.columns != "LABEL"]
    Y = data_train["LABEL"]
    classes, classes_counts = np.unique(Y, return_counts=True)

    # Bieu do phan bo so luong tung loai nhan trong dataset
    create_single_bar_plot(
        classes,
        classes_counts,
        "Classes",
        "Counts",
        "Class Distribution Of Dataset",
        "class_distribution.png",
    )

    return X, Y, classes


# Dem so luong tuong ung cua tung nhan
def count_unique_class(classes, data):
    count_dict = dict()
    for i in classes:
        count_dict[i] = 0
    value_counts, count = np.unique(data, return_counts=True)
    count_dict.update({A: B for A, B in zip(value_counts, count)})
    return count_dict


# Tao bang so sanh so luong tung nhan giua tap du lieu thuc te va tap du lieu du doan
def create_comparative_table(dict1, dict2, saved_name):
    df = pd.DataFrame()
    df["Class"] = dict1.keys()
    df["Test"] = dict1.values()
    df["Predict"] = dict2.values()
    dfi.export(df, "image_export/{}".format(saved_name), dpi=300)
    return df
