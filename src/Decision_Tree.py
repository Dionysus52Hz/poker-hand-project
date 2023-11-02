from handle_data import (
    create_single_bar_plot,
    create_comparative_table,
    count_unique_class,
    combine_image,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import random
import os

NUM_OF_EXECUTIONS = 10


def decision_tree(X, Y, classes):
    num_of_executions = list()
    result = list()
    for i in range(NUM_OF_EXECUTIONS):
        print("Lan lap thu {}.......".format(i + 1))
        num_of_executions.append(i + 1)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=1.0 / 3, random_state=random.randint(0, 100)
        )

        clf_gini = DecisionTreeClassifier(
            criterion="gini",
            random_state=100,
            max_depth=25,
            min_samples_leaf=4,
        )
        clf_gini.fit(X_train, Y_train)

        Y_predicted = clf_gini.predict(X_test)

        result.append(f1_score(Y_predicted, Y_test, average="macro"))

        if i == NUM_OF_EXECUTIONS - 1:
            test_class_counts_dict = count_unique_class(classes, Y_test)
            pred_class_counts_dict = count_unique_class(classes, Y_predicted)
            create_comparative_table(
                test_class_counts_dict,
                pred_class_counts_dict,
                "decision_tree_comparison_table.png",
            )
    print("Chi so F1-macro o lan chay thap nhat: ", min(result))
    print("Chi so F1-macro o lan chay cao nhat: ", max(result))
    print("Chi so F1-macro trung binh: ", sum(result) / NUM_OF_EXECUTIONS)

    create_single_bar_plot(
        num_of_executions,
        result,
        "Number of times the program has been executed",
        "F1-macro Score",
        "F1-macro score after 10 executions of the program using Decision-Tree Algorithm",
        "decision_tree_result.png",
    )

    image1 = Image.open("image_export/decision_tree_result.png")
    image2 = Image.open("image_export/decision_tree_comparison_table.png")
    combine_image(image1, image2, "decision_tree_result.png")
    os.remove("image_export/decision_tree_comparison_table.png")
