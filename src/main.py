from handle_data import preprocessing_data
from KNN import knn
from Bayes import bayes
from Decision_Tree import decision_tree
from Random_Forest import random_forest


def main():
    X, Y, classes = preprocessing_data()
    print("============== KNN ============")
    knn(X, Y, classes)
    print("============== Bayes ============")
    bayes(X, Y, classes)
    print("============== Cay Quyet Dinh ============")
    decision_tree(X, Y, classes)
    print("============== Rung Ngau Nhien ============")
    random_forest(X, Y, classes)


if __name__ == "__main__":
    main()
