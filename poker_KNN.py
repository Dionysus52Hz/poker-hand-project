from handle_data import split_data, create_bar_graph
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import random


X, Y = split_data()


run_times_list = list()
result_list = list()
for i in range(2):
    run_times_list.append(i)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random.randint(0, 100)
    )
    KNN = KNeighborsClassifier(n_neighbors=20)
    KNN.fit(X_train, Y_train)

    Y_predicted = KNN.predict(X_test)
    print(accuracy_score(Y_predicted, Y_test))

    # result_list.append(f1_score(Y_predicted, Y_test, average="macro"))

# create_bar_graph(
#     run_times_list,
#     result_list,
#     "Times",
#     "F1-macro Score",
#     "F1-macro score on 20 times hold-out validation",
#     "f1_score_knn.png",
# )
