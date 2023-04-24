<h1 align='center'>Stamox</h1>

[![PyPI version](https://badge.fury.io/py/stamox.svg)](https://badge.fury.io/py/stamox)
[![PyPI - License](https://img.shields.io/pypi/l/stamox)](https://pypi.org/project/stamox/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/stamox)](https://pypi.org/project/stamox/)
[![GitHub stars](https://img.shields.io/github/stars/jiayaobo/stamox)]()

# Hello Stamox, Why Another Wheel?


# Why Another Wheel?

What stamox does is really simple, just make it possible, it aims to provide a statistic interface for `JAX`. But nowadays, we have so many statistic packages around the world varying from languages, for python, `statsmodels` just works great, for `R`, `tidyverse` derived packages are so delicate and easy to use. So **why build another wheel**?

Three reasons I think:

* Personal interest, as a student of statistics, I want to learn more about statistics and machine learning, proficient knowledge comes from books but more from practice, write down the code behind the theory is a good way to learn.

* Speed, `JAX` is really fast, and `Equinox` is a good tool to make `JAX` more convenient, backend of `JAX` is `XLA`, which makes it possible to compile the code to GPU or TPU, and it is really fast.

* Easy of Use, `%>%` is delicate operation in `R`, it combines the functions to a pipe and make the code more readable, and `stamox` is inspired by it, and I want to take a try to make it convenient in python with `>>`.

And here're few benchmarks:

*generate random variables*

![benchmark](./benchmark/benchmark1.png)

*calculate cdf*

![benchmark](./benchmark/benchmark2.png)

## Installation

```bash
pip install -U stamox
# or
pip install git+[stamox](https://github.com/JiaYaobo/stamox.git)
```

## Documentation

More comprehensive introduction and examples can be found in the [documentation](https://jiayaobo.github.io/stamox/).

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

`>>` is the pipe operator, which is the similar to `|>` in `F#` and `Elixir` or `%>%` in `R`.

* Internal Functions Pipeable

```python
import jax.random as jrandom
from stamox import pipe_jit
from stamox.basic import scale
from stamox.distribution import rnorm
from stamox.regression import lm

key = jrandom.PRNGKey(20010813)

@pipe_jit
def f(x):
    return [3 * x[:, 0] + 2 * x[:, 1] - x[:, 2], x] # [y, X]
pipe = rnorm(sample_shape=(1000, 3)) >> f >> lm
state = pipe(key)
print(state.params)
```

### Linear Regression with Formula

```python
import pandas as pd
import numpy as np
from stamox.regression import lm


x = np.random.uniform(size=(1000, 3))
y = 2 * x[:,0] + 3 * x[:,1] + 4 * x[:,2] + np.random.normal(size=1000)
df = pd.DataFrame(x, columns=['x1', 'x2', 'x3'])
df['y'] = y

lm(df, 'y~x1+x2+x3').params
```

* Custom Functions Pipeable

```python
from stamox import make_pipe, make_partial_pipe, Pipeable
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

You can use autograd features from `JAX` and `Equinox` with `Stamox` easily.

```python
from stamox import make_pipe, make_partial_pipe, Pipeable
import jax.numpy as jnp
from equinox import filter_jit, filter_vmap, filter_grad

@make_partial_pipe
@filter_jit
@filter_vmap
@filter_grad
def f(x, y):
    return y * x ** 3
       
f(y=3.)(jnp.array([1., 2., 3.]))
```

Or vmap, pmap, jit features integrated with `Stamox`:

```python
from stamox import pipe_vmap, pipe_jit

@pipe_vmap
@pipe_jit
def f(x):
    return x ** 2

f(jnp.array([1., 2., 3.]))
```

## Acceleration Support

`JAX` can be accelerated by `GPU` and `TPU`. So, `Stamox` is compatible with them.

## See More

[JAX](https://github.com/google/jax)

[Equinox](https://github.com/patrick-kidger/equinox#readme)