from handle_data import split_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random


X, Y = split_data()

for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random.randint(0, 100)
    )
    KNN = KNeighborsClassifier(n_neighbors=157)
    KNN.fit(X_train, Y_train)

    Y_predicted = KNN.predict(X_test)

    print(accuracy_score(Y_test, Y_predicted) * 100)
