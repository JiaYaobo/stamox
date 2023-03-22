import functools as ft

import jax.numpy as jnp
from jax import jit

from stamox.distribution import pnorm, qnorm

def p_test(x, mu=0, alternative="two.sided", conf_level=0.95):
    return _p_test_single(x, mu, alternative, conf_level)

@ft.partial(jit, static_argnames=('alternative', 'mu', 'conf_level',))
def _p_test_single(x, mu=0, alternative="two.sided", conf_level=0.95):
    nx = x.shape[0]
    mx = jnp.mean(x, axis=-1, keepdims=True)
    vx = jnp.var(x, axis=-1, keepdims=True)
    stderr = jnp.sqrt(vx/nx)
    z_stat = (mx - mu) / stderr
    pval = 2 * pnorm(-jnp.abs(z_stat))
    alpha = jnp.array([1. - conf_level], dtype=x.dtype)
    conf_int  = qnorm(1 - alpha / 2)
    conf_int = jnp.array([-conf_int, conf_int]) + z_stat
    conf_int = conf_int * stderr + mu

    return z_stat, pval, conf_int