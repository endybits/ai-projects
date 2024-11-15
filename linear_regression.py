import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1500):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            self._gradient_descent(X, y, n_samples)

    def _gradient_descent(self, X: np.ndarray, y, n_samples):
        y_predict = np.dot(X, self.weights) + self.bias

        dw = (1/n_samples) * np.dot(X.T, (y_predict - y))
        db = (1/n_samples) * np.sum(2 * (y_predict - y))

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db


    def predict(self, X: np.ndarray):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred