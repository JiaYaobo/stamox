import numpy as np


class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        # Add column of ones for bias term
        X = np.insert(X, 0, 1, axis=1)

        # Initialize coefficients to zeros
        self.coef_ = np.zeros(X.shape[1])

        # Iterate until convergence or maximum iterations reached
        for i in range(self.max_iter):
            prev_coef = np.copy(self.coef_)

            # Calculate gradient of loss function
            grad = self._lasso_grad(X, y, self.coef_)

            # Update coefficients using gradient descent
            self.coef_ -= grad

            # Apply soft thresholding to enforce L1 penalty
            self.coef_ = np.sign(self.coef_) * np.maximum(
                np.abs(self.coef_) - self.alpha, 0.0
            )

            # Check for convergence
            if np.linalg.norm(self.coef_ - prev_coef) < self.tol:
                break

    def predict(self, X):
        # Add column of ones for bias term
        X = np.insert(X, 0, 1, axis=1)

        # Predict target variable using learned coefficients
        return np.dot(X, self.coef_)

    def _lasso_grad(self, X, y, coef):
        # Compute gradient of L1-regularized least squares loss function
        error = np.dot(X, coef) - y
        grad = np.dot(X.T, error)
        grad[:-1] += self.alpha * np.sign(coef[:-1])
        return grad
