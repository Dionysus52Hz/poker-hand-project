from handle_data import (
    create_single_bar_plot,
    create_comparative_table,
    count_unique_class,
    combine_image,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import random
import os


NUM_OF_EXECUTIONS = 10


def bayes(X, Y, classes):
    num_of_executions = list()
    result = list()
    for i in range(NUM_OF_EXECUTIONS):
        print("Lan lap thu {}.......".format(i + 1))
        num_of_executions.append(i + 1)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=1.0 / 3, random_state=random.randint(0, 100)
        )

        bayes_model = GaussianNB()
        bayes_model.fit(X_train, Y_train)

        Y_predicted = bayes_model.predict(X_test)
        result.append(f1_score(Y_predicted, Y_test, average="macro"))

        # Tao bang so sanh o lan chay cuoi cung
        if i == NUM_OF_EXECUTIONS - 1:
            test_class_counts_dict = count_unique_class(classes, Y_test)
            pred_class_counts_dict = count_unique_class(classes, Y_predicted)
            create_comparative_table(
                test_class_counts_dict,
                pred_class_counts_dict,
                "bayes_comparison_table.png",
            )

    create_single_bar_plot(
        num_of_executions,
        result,
        "Number of times the program has been executed",
        "F1-macro Score",
        "F1-macro score after 10 executions of the program using Bayes Algorithm",
        "bayes_result.png",
    )

    image1 = Image.open("image_export/bayes_result.png")
    image2 = Image.open("image_export/bayes_comparison_table.png")
    combine_image(image1, image2, "bayes_result.png")
    os.remove("image_export/bayes_comparison_table.png")
