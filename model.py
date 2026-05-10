import numpy as np

class Adaline:
    # 0.01 learning rate by default
    def __init__(self, n_features, lr=1e-2):
        self.w = np.zeros(n_features)
        self.b = 0
        self.lr = lr
        self.errors = []
        print(f"Zainicjalizowano Adaline z {n_features} cechami, lr={lr}")

    # linear activation function
    def activation(self, X):
        return np.dot(X, self.w) + self.b

    # training using batch gradient descent
    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            output = self.activation(X)
            error = y - output

            N = X.shape[0]
            # how much each pixel contributes to the error
            self.w += self.lr * np.dot(X.T, error) / N
            self.b += self.lr * error.sum() / N

            mse = (error ** 2).mean()
            self.errors.append(mse)
            
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoka {epoch + 1}/{epochs}, MSE: {mse:.6f}")

    def predict(self, X):
        return self.activation(X)