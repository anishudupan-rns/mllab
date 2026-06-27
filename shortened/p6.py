import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
x_test = np.linspace(0, 2 * np.pi, 200)
x_bias = np.c_[np.ones(100), X]
y = np.sin(X) + 0.1 * np.random.randn(100)
print("x_test",x_test)


def predict(xi, tau=0.5):
    W = np.diag(np.exp(-np.sum((x_bias - xi) ** 2, axis=1) / (2 * tau**2)))
    return xi @ (np.linalg.inv(x_bias.T @ W @ x_bias) @ x_bias.T @ W @ y)


# Execution & Visualization
y_pred = [predict([1, x]) for x in x_test]

plt.scatter(X, y, alpha=0.7, label="Training Data")
plt.plot(x_test, y_pred, lw=2, label="LWR Fit")
plt.legend()
plt.title("Locally Weighted Regression")
plt.show()
