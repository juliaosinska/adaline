import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# 70 thousands samples, 784 features (28x28 pixels), 10 classes (digits 0-9)
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # X shape 70000 rows, 784 columns
    X = mnist.data
    # labels
    y = mnist.target.astype(int)

    x_train, x_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test

def prepare_ova_labels(y, n_classes=10):
    Y = np.zeros((n_classes, len(y)))
    for i in range(n_classes):
        Y[i] = np.where(y == i, 1, -1)
    return Y