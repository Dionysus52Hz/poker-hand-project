from handle_data import split_data, create_bar_graph
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import random

X, Y = split_data()


run_times_list = list()
result_list = list()
# for i in range(20):
#     run_times_list.append(i + 1)
#     X_train, X_test, Y_train, Y_test = train_test_split(
#         X, Y, test_size=0.2, random_state=random.randint(0, 100)
#     )
#     bayes_model = GaussianNB()
#     bayes_model.fit(X_train, Y_train)
#     Y_predicted = bayes_model.predict(X_test)
#     result_list.append(accuracy_score(Y_predicted, Y_test) * 100)
#     print(accuracy_score(Y_predicted, Y_test) * 100)
k = 1
skf = KFold(n_splits=20)
for train_idx, test_idx in skf.split(X, Y):
    run_times_list.append(k)
    k += 1
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    bayes_model = GaussianNB()
    bayes_model.fit(X_train, Y_train)
    Y_predicted = bayes_model.predict(X_test)
    result_list.append(f1_score(Y_predicted, Y_test, average="macro"))


create_bar_graph(
    run_times_list,
    result_list,
    "Fold",
    "F1-macro Score",
    "F1-macro score on 20-fold validation",
    "f1_score_bayes.png",
)
