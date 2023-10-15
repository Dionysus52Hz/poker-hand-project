from handle_data import split_data, create_bar_graph
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
import random

X, Y = split_data()
run_times_list = list()
result_list = list()
for i in range(20):
    run_times_list.append(i + 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random.randint(0, 100)
    )
    clf_gini = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=25, min_samples_leaf=10
    )
    clf_gini.fit(X_train, Y_train)
    Y_predicted = clf_gini.predict(X_test)
    result_list.append(f1_score(Y_predicted, Y_test, average="macro"))

create_bar_graph(
    run_times_list,
    result_list,
    "Times",
    "F1-macro Score",
    "F1-macro score on 20 times hold-out validation",
    "f1_score_dt1.png",
)


run_times_list.clear()
result_list.clear()
k = 1
skf = StratifiedKFold(n_splits=3)
for train_idx, test_idx in skf.split(X, Y):
    run_times_list.append(k)
    k += 1
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    clf_gini = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=25, min_samples_leaf=10
    )
    clf_gini.fit(X_train, Y_train)
    Y_predicted = clf_gini.predict(X_test)
    result_list.append(f1_score(Y_predicted, Y_test, average="macro"))

create_bar_graph(
    run_times_list,
    result_list,
    "Folds",
    "F1-macro Score",
    "F1-macro score on 8-fold validation",
    "f1_score_dt2.png",
)
