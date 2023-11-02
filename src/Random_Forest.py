from handle_data import (
    count_unique_class,
    create_single_bar_plot,
    create_comparative_table,
    combine_image,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random
import os

# Số lần chạy hold-out
NUM_OF_EXECUTIONS = 10


def random_forest(X, Y, classes):
    num_of_executions = list()  # Lưu số lần chạy
    result = list()  # Lưu chỉ số f1 sau mỗi lần chạy
    for i in range(NUM_OF_EXECUTIONS):
        print("Lan lap thu {}.......".format(i + 1))
        num_of_executions.append(i + 1)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=1.0 / 3, random_state=random.randint(0, 100)
        )

        rf = RandomForestClassifier(
            n_estimators=51, random_state=random.randint(0, 100), n_jobs=4
        )
        rf.fit(X_train, Y_train)

        Y_predicted = rf.predict(X_test)

        # Do tập dữ liệu rất mất cân bằng nên sử dụng chỉ số f1 macro để đánh giá mô hình
        result.append(f1_score(Y_predicted, Y_test, average="macro"))

        # Vẽ bảng so sánh kết quả giữa tập test và tập dự đoán ở lần chạy cuối cùng
        if i == NUM_OF_EXECUTIONS - 1:
            test_class_counts_dict = count_unique_class(classes, Y_test)
            pred_class_counts_dict = count_unique_class(classes, Y_predicted)
            create_comparative_table(
                test_class_counts_dict,
                pred_class_counts_dict,
                "random_forest_comparison_table.png",
            )

            # Vẽ biểu đồ đánh giá độ ảnh hưởng của các thuộc tính đến kết quả dự đoán
            feature_imp = pd.Series(
                rf.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
            sb.barplot(x=feature_imp, y=feature_imp.index)
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.title("Feature importances")
            plt.gcf().set_size_inches(1280 / 150, 720 / 150)
            plt.savefig("image_export/feature_importances.png", dpi=150)
            plt.clf()

    print("Chi so F1-macro o lan chay thap nhat: ", min(result))
    print("Chi so F1-macro o lan chay cao nhat: ", max(result))
    print("Chi so F1-macro trung binh: ", sum(result) / NUM_OF_EXECUTIONS)

    # Vẽ biểu đồ thể hiện độ chính xác mô hình sau tất cả lần chạy
    create_single_bar_plot(
        num_of_executions,
        result,
        "Number of times the program has been executed",
        "F1-macro Score",
        "F1-macro score after 10 executions of the program using Random-Forest Algorithm",
        "random_forest_result.png",
    )

    # Gộp biểu đồ độ chính xác và bảng so sánh lại thành 1 ảnh cho tiện
    image1 = Image.open("image_export/random_forest_result.png")
    image2 = Image.open("image_export/random_forest_comparison_table.png")
    combine_image(image1, image2, "random_forest_result.png")
    os.remove("image_export/random_forest_comparison_table.png")
