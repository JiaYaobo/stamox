import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit

from sax.distrix import pgamma, qgamma, rgamma
