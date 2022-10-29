import functools as ft

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from numpyro.distributions import StudentT

@ft.partial(jit, static_names=('alternative', 'mu', 'conf_level'))
def _t_test_single(x, alternativae="two.sided", mu=0, conf_level=0.95):
    nx = x.shape[0]
    mx = jnp.mean(x, axis=-1)
    vx = jnp.var(x, axis=-1)

    df = nx - 1
    stderr = jnp.sqrt(vx/nx)

    t_stat = (mx - mu) / stderr

    t_dist = StudentT(df=df)

    pval = 2 * t_dist.cdf(-jnp.abs(t_stat))

    alpha = 1. - conf_level

    conf_int  = t_dist.icdf(1 - alpha / 2)

    conf_int = jnp.array([-conf_int, conf_int]) + t_stat

    conf_int = conf_int * stderr + mu


    return t_stat, pval, conf_int




def _t_test_pair(x, y, alternativate="two.sided", mu=0, conf_level=0.95):
    pass

