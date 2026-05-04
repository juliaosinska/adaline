import numpy as np
from model import Adaline

def train_models(X, Y, n_classes=10, epochs=10):
    models = []

    for i in range(n_classes):
        print(f"Trening klasy {i}")
        model = Adaline(n_features=X.shape[1])
        model.train(X, Y[i], epochs=epochs)
        models.append(model)

    return models

def predict(models, X):
    outputs = np.array([m.predict(X) for m in models])
    return np.argmax(outputs, axis=0)

def evaluate(y_pred, y_true):
    acc = (y_pred == y_true).mean()
    return acc, 1 - acc