from functools import partial

import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import filter_jit
from jax import jit

from ...core import make_partial_pipe
from ...distribution import pF
from ...hypothesis import HypoTest


class OneWayAnovaTest(HypoTest):
    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
        name="One Way Anova Test",
    ):
        super().__init__(
            statistic, parameters, p_value, estimate, null_value, alternative, name
        )


@partial(jit, static_argnames=("axis"))
def _square_of_sums(a, axis=0):
    s = jnp.sum(a, axis, keepdims=True)
    return s * s


@partial(jit, static_argnames=("axis"))
def _sum_of_squares(a, axis=0):
    return jnp.sum(a * a, axis=axis, keepdims=True)


@make_partial_pipe
def one_way(*samples, axis=0) -> OneWayAnovaTest:
    samples = [jnp.asarray(sample) for sample in samples]
    ngroups = len(samples)
    return _one_way(samples, ngroups, axis=axis)


@filter_jit
def _one_way(samples, ngroups, axis=0):
    alldata = jnp.concatenate(samples, axis=axis)
    N = alldata.shape[axis]
    offset = jnp.mean(alldata, axis=axis, keepdims=True)
    alldata -= offset
    normalized_ss = _square_of_sums(alldata, axis=axis) / N

    # total
    sstot = _sum_of_squares(alldata, axis=axis) - normalized_ss

    # between groups or namely treatments
    ssbn = jnp.sum(
        jnp.concatenate(
            jtu.tree_map(
                lambda s: _square_of_sums(s - offset, axis) / s.shape[0], samples
            ),
            axis,
        ),
        axis,
        keepdims=True,
    )
    ssbn -= normalized_ss

    # with in groups
    sswn = sstot - ssbn

    # degree of freedom
    dfbn = ngroups - 1
    dfwn = N - ngroups

    # mean squares
    msb = ssbn / dfbn
    msw = sswn / dfwn

    # F-ratio
    f = msb / msw
    prob = 1 - pF(f, dfbn, dfwn)
    return OneWayAnovaTest(statistic=f, p_value=prob)
