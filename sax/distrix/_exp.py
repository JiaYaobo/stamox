import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit
from jax.scipy.special import gammainc
import  tensorflow_probability.substrates.jax.math as tfp_math

