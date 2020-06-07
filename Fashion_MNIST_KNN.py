import mnist_reader
import numpy as np
import matplotlib.pyplot as plt


def manhattan_distance(X, X_train):
    manhattan_dist_tab = np.zeros((len(X), len(X_train)))
    for i in range(0, len(X)):
        print(i)
        dist = abs(X_train - X[i])
        manhattan_dist_tab[i] = np.sum(dist, axis=1)
    return manhattan_dist_tab


def sort_train_labels_knn(Dist, y):
    order = np.argsort(Dist, kind='mergesort', axis=1)
    return y[order]


def p_y_x_knn(y, k):
    classes = [i for i in range(10)]
    result = []
    [result.append([np.sum(row[0:k] == x) for x in classes]) for row in y]
    return np.array(result) / k


def classification_error(p_y_x, y_true):
    predict_labels = np.argmax(p_y_x, axis=1)
    result = np.mean(y_true != predict_labels)
    return result


def show_diagram(errors):
    plt.plot(range(1, 30), 1 - np.array(errors), 'g')
    plt.legend(('Accuracy', ''))
    plt.ylabel('Accuracy')
    plt.xlabel('KNN')
    plt.show()


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    Xval = X_val.astype(int)
    Xtrain = X_train.astype(int)
    yval = y_val.astype(int)
    ytrain = y_train.astype(int)

    dist = manhattan_distance(Xval, Xtrain)
    sorted_labels = sort_train_labels_knn(dist, ytrain)
    errors = [classification_error(p_y_x_knn(sorted_labels, k), yval) for k in k_values]

    best_error = np.amin(errors)
    best_k = k_values[np.argmin(errors)]
    print(f"Jakość modelu KNN to: {(1 - best_error)*100}% , najlepsza ilość sąsiadów to: {best_k}")
    show_diagram(errors)


X_train, y_train = mnist_reader.load_mnist('fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion', kind='t10k')
k = [i for i in range(1, 30)]
model_selection_knn(X_test, X_train, y_test, y_train, k)
