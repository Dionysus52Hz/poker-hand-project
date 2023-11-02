import pandas as pd
import numpy as np
import dataframe_image as dfi
from matplotlib import pyplot as plt
from PIL import Image


# Hàm đếm số lượng của từng nhãn có trong tập dữ liệu
def count_unique_class(classes, data):
    count_dict = dict()
    for i in classes:
        count_dict[i] = 0
    value_counts, count = np.unique(data, return_counts=True)
    count_dict.update({A: B for A, B in zip(value_counts, count)})
    return count_dict


# Hàm tạo biểu đồ cột
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
    plt.gcf().set_size_inches(1280 / 150, 720 / 150)
    plt.savefig("image_export/{}".format(saved_name), dpi=150)
    plt.clf()


# Hàm tạo bảng so sánh số lượng từng nhãn giữa tập dữ liệu test và tập dữ liệu dự đoán
def create_comparative_table(dict1, dict2, saved_name):
    df = pd.DataFrame()
    df["Class"] = dict1.keys()
    df["Test"] = dict1.values()
    df["Predict"] = dict2.values()
    dfi.export(df, "image_export/{}".format(saved_name), dpi=200)


# Hàm sắp xếp 5 lá bài của mỗi dòng dữ liệu theo thứ tự tăng dần
def sort_rank(data):
    df = data.copy()
    dfc = df[["C1", "C2", "C3", "C4", "C5"]]
    dfc.values.sort()
    df[["C1", "C2", "C3", "C4", "C5"]] = dfc
    df = df[["C1", "C2", "C3", "C4", "C5", "S1", "S2", "S3", "S4", "S5", "LABEL"]]
    return df


# Hàm thêm vào tập dữ liệu ban đầu 1 cột mới chứa số lá bài phân biệt tương ứng với mỗi dòng
def add_unique_count(data):
    tmp = data[["S1", "S2", "S3", "S4", "S5"]]
    data["Unique Suits"] = tmp.apply(lambda x: len(np.unique(x)), axis=1)


# Hàm thêm vào tập dữ liệu ban đầu các cột chứa độ chênh lệch giữa các cặp 2 lá bài liền kề nhau
def add_differences_between_ranks(data):
    data["Diff1"] = data["C2"] - data["C1"]
    data["Diff2"] = data["C3"] - data["C2"]
    data["Diff3"] = data["C4"] - data["C3"]
    data["Diff4"] = data["C5"] - data["C4"]


# Tiền xử lí dữ liệu
def preprocessing_data():
    # Thêm tên cho các cột trong tập dữ liệu
    features = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "LABEL"]
    data = pd.read_csv("dataset/poker-hand-testing.data", delimiter=",", names=features)
    data = sort_rank(data)
    X = data.loc[:, data.columns != "LABEL"]
    Y = data["LABEL"]
    add_unique_count(X)
    add_differences_between_ranks(X)

    # Biểu đồ biểu thị sự phân chia số lượng của từng nhãn trong tập dữ liệu gốc
    classes, classes_counts = np.unique(Y, return_counts=True)
    create_single_bar_plot(
        classes,
        classes_counts,
        "Classes",
        "Counts",
        "Class Distribution Of Dataset",
        "class_distribution.png",
    )
    return X, Y, classes


# Hàm ghép 2 ảnh lại thành 1
def combine_image(image1, image2, saved_name):
    width1, height1 = image1.size
    width2, height2 = image2.size
    new_width = width1 + width2 + 30
    new_height = max(height1, height2)
    combined_image = Image.new("RGB", (new_width, new_height), color="white")
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 20))
    combined_image.save("image_export/{}".format(saved_name))
