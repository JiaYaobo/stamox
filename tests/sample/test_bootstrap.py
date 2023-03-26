import jax.random as jrandom

import jax.numpy as jnp

from stamox.core import Pipeable
from stamox.sample import bootstrap_sample
from stamox.basic import mean



key = jrandom.PRNGKey(0)

x = jrandom.normal(key=key, shape=(10, 10))



h = Pipeable(x)



print(id(h.value))
print(id(x))