# Examples

## Basic Fucntions

```python
import jax.numpy as jnp
from stamox.basic import mean, var, std
from stamox.core import Pipeable

x = jnp.ones((3, 4))
mean(x) # 1.0
var(x) # 0.0
std(x) # 0.0

# Pipeable
f = Pipeable(x) >> mean(axis=0) >> var
f() # 0.0
```

## Distributions

```python
import jax.random as jrandom
from stamox.distribution import pnorm, rnorm, qnorm, dnorm

key = jrandom.PRNGKey(20010813)
x = rnorm(key, sample_shape=(8, 100000))
cdf = pnorm(x)
q = qnorm(cdf)
pdf = dnorm(q)
```

## Linear Model

```python
import jax.numpy as jnp
from stamox.regression import lm

X = np.random.uniform(size=(1000, 3))
y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
data = pd.DataFrame(
    np.concatenate([X, y.reshape((-1, 1))], axis=1),
    columns=["x1", "x2", "x3", "y"],
)

res = lm(data, "y ~ x1 + x2 + x3")
# or
res  = (Pipeable(data) >> lm("y ~ x1 + x2 + x3"))()
```
```

