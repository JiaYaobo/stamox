import functools as ft

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from stamox.distrix import pt, qt


def t_test(x, mu=0, alternative="two.sided", conf_level=0.95):
    return _t_test_single(x, mu, alternative, conf_level)


@ft.partial(
    jit,
    static_argnames=(
        "alternative",
        "mu",
        "conf_level",
    ),
)
def _t_test_single(x, mu=0, alternative="two.sided", conf_level=0.95):
    nx = x.shape[0]
    mx = jnp.mean(x, axis=-1, keepdims=True)
    vx = jnp.var(x, axis=-1, keepdims=True)

    df = nx - 1
    stderr = jnp.sqrt(vx / nx)
    t_stat = (mx - mu) / stderr
    pval = 2 * pt(-jnp.abs(t_stat), df)
    alpha = jnp.array([1.0 - conf_level], dtype=x.dtype)
    conf_int = qt(1 - alpha / 2, df)
    conf_int = jnp.array([-conf_int, conf_int]) + t_stat
    conf_int = conf_int * stderr + mu

    return t_stat, pval, conf_int


def _t_test_pair(x, y, alternativate="two.sided", mu=0, conf_level=0.95):
    pass
