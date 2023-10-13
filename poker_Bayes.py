from handle_data import split_data
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import random

X, Y = split_data()

run_times_list = list()
result_list = list()
for i in range(20):
    run_times_list.append(i + 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random.randint(0, 100)
    )
    bayes_model = GaussianNB()
    bayes_model.fit(X_train, Y_train)
    Y_predicted = bayes_model.predict(X_test)
    result_list.append(accuracy_score(Y_predicted, Y_test) * 100)
    print(accuracy_score(Y_predicted, Y_test) * 100)

plt.bar(run_times_list, result_list, width=0.75)
plt.xlabel("Run Times")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Score after running {} times".format(len(run_times_list)))
plt.xticks(run_times_list)
plt.ylim(0, 100)
plt.tight_layout()
for i in range(len(result_list)):
    plt.text(
        run_times_list[i],
        result_list[i],
        str(round(result_list[i], 2)),
        ha="center",
        va="bottom",
    )
plt.show()
