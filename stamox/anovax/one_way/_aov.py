import functools as ft

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import jit

from stamox.distrix import pF


@ft.partial(jit, static_argnames=('axis'))
def _square_of_sums(a, axis=0):
    s = jnp.sum(a, axis, keepdims=True)
    return s * s

@ft.partial(jit, static_argnames=('axis'))
def _sum_of_squares(a, axis=0):
    return jnp.sum(a * a, axis=axis, keepdims=True)


def one_way(*samples, axis=0):
    samples = [jnp.asarray(sample) for sample in samples]
    ngroups = len(samples)
    return _one_way(samples,  ngroups, axis=axis)

@ft.partial(jit, static_argnames=('ngroups', 'axis', ))
def _one_way(samples, ngroups, axis=0):
    alldata = jnp.concatenate(samples, axis=axis)
    N = alldata.shape[axis]
    offset = jnp.mean(alldata, axis=axis, keepdims=True)
    alldata -= offset
    normalized_ss = _square_of_sums(alldata, axis=axis) / N

    # total
    sstot = _sum_of_squares(alldata, axis=axis) - normalized_ss

    # between groups or namely treatments
    ssbn = jnp.sum(jnp.concatenate(jtu.tree_map(lambda s : _square_of_sums(s - offset, axis) / s.shape[0], samples), axis), axis, keepdims=True)
    ssbn -= normalized_ss

    # with in groups
    sswn = sstot - ssbn

    # degree of freedom
    dfbn = ngroups - 1
    dfwn = N - ngroups

    # mean squares
    msb = ssbn / dfbn
    msw = sswn / dfwn

    #F-ratio
    f = msb / msw
    prob = 1 - pF(f, dfbn, dfwn)
    return f, prob