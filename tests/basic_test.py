import jax.numpy as jnp
from stamox.distribution import pnorm, dnorm
from stamox.basic import mean



print(pnorm(mean=0.5, sd=2)(jnp.array([1., 2.])))
