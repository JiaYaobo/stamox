import numpy as np
from stamox import regression
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from stamox.core import Pipeable

key = jrandom.PRNGKey(0)
X = jrandom.uniform(key, shape=(1000, 2))
y = 3 * X[:, 0] + 2 * X[:, 1]  + 1.
y = y.reshape((-1,1))

ols = regression.OLS(key=key)


h = Pipeable(jnp.hstack([X, y])) >> ols

print(h().params)