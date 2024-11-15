import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from linear_regression import LinearRegression


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)




l_reg = LinearRegression(learning_rate=0.01, n_iters=2000)

l_reg.fit(X_train, y_train)

predictions = l_reg.predict(X_test)


def mse(y_test, predictions):
    return (np.sum((y_test - predictions) ** 2)) / len(y_test)


mse_value = mse(y_test, predictions)

print("MSE:", mse_value)


y_predict_line = l_reg.predict(X)

c_map = plt.get_cmap("viridis")

fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=c_map(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=c_map(0.5), s=10)

plt.plot(X, y_predict_line, color="black", linewidth=2, label="Prediction")
plt.show()