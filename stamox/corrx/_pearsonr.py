import functools as ft

import jax.numpy as jnp
from jax import jit
from stamox.distrix import pbeta


def pearsonr(x, y, alternative='two-sided'):
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    return _pearsonr(x, y, alternative)


@ft.partial(jit, static_argnames=('alternative', ))
def _pearsonr(x, y, alternative='two-sided'):
    n = x.size

    xm = x - x.mean()
    ym = y - y.mean()

    normxm = jnp.linalg.norm(xm)
    normym = jnp.linalg.norm(ym)

    r = jnp.dot(xm/normxm, ym/normym)

    r = jnp.maximum(jnp.minimum(r, 1.0), -1.0)
    ab = n / 2 - 1

    if alternative == 'two-sided':
        prob = 2*pbeta(0.5*(1 - jnp.abs(r)), ab, ab)
    elif alternative == 'less':
        prob = 1 - pbeta(0.5*(1 - jnp.abs(r)), ab, ab)
    else:
        prob = pbeta(0.5*(1 - jnp.abs(r)), ab, ab)

    return r, prob
