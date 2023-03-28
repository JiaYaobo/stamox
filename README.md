<h1 align='center'>Stamox</h1>

# Stamox: A Thin Wrapper of `JAX` and `Equinox` for Statistics

Just out of curiosity of Fucntional Programming, I wrote this package. It is a thin wrapper of `JAX` and `Equinox` for statistics. It is not a complete package, and in heavy development.

Inspired by many packages from Python and R, I hope to fuse different features of them into one package, like `%>%` in `dplyr`,
or apis from statsmodels and scipy etc.


## Installation

```bash
pip install stamox
```

## Documentation

Not yet.

## Quick Start

### Similar but faster distribution functions to `R`

```python
from stamox.distribution import *
import jax.random as jrandom

key = jrandom.PRNGKey(20010813)

# random
x = rnorm(key, sample_shape=(1000, ))
# cdf
pnorm(x)
# ppf
qnorm(x)
# pdf
dnorm(x)
```

### Fearless Pipeable

`>>` is the pipe operator, which is the similar to `|>` in `F#` and `Elixir` or `%>%` in `R`. But `>>` focus on the composition of functions not the data. You must call pipeable functions with `()`.

* Internal Functions Pipeable

```python
from stamox.core import make_pipe
from stamox.regression import OLS
from stamox.distribution import rnorm
from stamox.basic import scale
from equinox import filter_jit
import jax.random as jrandom

key = jrandom.PRNGKey(20010813)

@make_pipe
@filter_jit
def f(x):
    return [x, 3 * x[:, 0] + 2 * x[:, 1] - x[:, 2]]
pipe = rnorm(sample_shape=(1000, 3)) >> scale >> f >> OLS(use_intercept=False, key=key)
state = pipe(key)
print(state.params)
```



* Custom Functions Pipeable

```python
from stamox.core import make_pipe, make_partial_pipe, Pipeable
import jax.numpy as jnp
import jax.random as jrandom

x = jnp.ones((1000, ))
# single input, simply add make pipe
@make_pipe
def f(x):
    return x ** 2

# multiple input, add make partial pipe
@make_partial_pipe
def g(x, y):
    return x + y

# Notice Only One Positional Argument Can Be Received Along the pipe
h = Pipeable(x) >> f >> g(y=2.) >> f >> g(y=3.) >> f
print(h())
```

* Compatible With `JAX` and `Equinox`

```python
from stamox.core import make_pipe, make_partial_pipe, Pipeable
import jax.numpy as jnp
from equinox import filter_jit, filter_vmap, filter_grad

@make_partial_pipe
@filter_jit
@filter_vmap
@filter_grad
def f(x, y):
    return y * x ** 3
       
print(f(y=3.)(jnp.array([1., 2., 3.])))
```

## Acceleration Support

`JAX` can be accelerated by `GPU` and `TPU`. `Stamox` is compatible with them.

## See More

[JAX](https://github.com/google/jax)

[Equinox](https://github.com/patrick-kidger/equinox#readme)