import numpy as np


def hypothesis(X, theta):
    return X @ theta


def compute_cost(X, y, theta):
    residuals = hypothesis(X, theta) - y
    return float(residuals @ residuals / (2 * len(y)))


def normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def batch_gradient_descent(X, y, alpha, iterations, theta=None):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    theta = np.zeros(X.shape[1]) if theta is None else np.array(theta, dtype=float)
    if theta.shape != (X.shape[1],):
        raise ValueError("theta must have one value per design-matrix column")

    history = []
    for _ in range(iterations):
        error = hypothesis(X, theta) - y
        gradient = X.T @ error / len(y)
        theta = theta - alpha * gradient
        history.append(compute_cost(X, y, theta))
    return theta, history


def fit_sgd(X, y, alpha, epochs, seed=42):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    theta = np.zeros(X.shape[1])
    rng = np.random.default_rng(seed)
    history = []

    for _ in range(epochs):
        for index in rng.permutation(len(y)):
            error = hypothesis(X[index], theta) - y[index]
            theta -= alpha * X[index] * error
        history.append(compute_cost(X, y, theta))
    return theta, history


def compute_rmse(y_true, y_pred):
    if len(y_true) == 0:
        raise ValueError("RMSE cannot be calculated for an empty target array")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
