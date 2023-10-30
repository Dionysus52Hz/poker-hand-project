from handle_data import (
    split_data,
    create_single_bar_plot,
    create_comparative_table,
    count_unique_class,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import random

NUM_OF_EXECUTIONS = 10

X, Y, classes = split_data()


num_of_executions = list()
result = list()

for i in range(20):
    print("Lan lap thu {}.......".format(i + 1))
    num_of_executions.append(i + 1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1.0 / 3, random_state=random.randint(0, 100)
    )
    KNN = KNeighborsClassifier(n_neighbors=999)
    KNN.fit(X_train, Y_train)

    Y_predicted = KNN.predict(X_test)
    print(f1_score(Y_test, Y_predicted, average="macro"))
    result.append(f1_score(Y_test, Y_predicted, average="macro"))

    if i == NUM_OF_EXECUTIONS - 1:
        test_class_counts_dict = count_unique_class(classes, Y_test)
        pred_class_counts_dict = count_unique_class(classes, Y_predicted)
        create_comparative_table(
            test_class_counts_dict,
            pred_class_counts_dict,
            "knn_comparison_table.png",
        )

create_single_bar_plot(
    num_of_executions,
    result,
    "Number of times the program has been executed",
    "F1-macro Score",
    "F1-macro score after 10 executions of the program using KNN algorithm",
    "knn_result.png",
)
