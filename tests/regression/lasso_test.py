import numpy as np
from stamox import regression

lasso = regression.LassoRegression(alpha=0.1, tol=0.0001)


# Generate some test data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])


lasso.fit(X, y)

print(lasso.coef_)