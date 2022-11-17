import functools as ft
import warnings

import jax
import jax.numpy as jnp
from jax import jit
from scipy.stats import rankdata as rank_scipy


@ft.partial(jit, static_argnames=('size', ))
def _rank_dense_avg(x, size):
    arr = x
    sorter = jnp.argsort(arr)
    arr = arr[sorter]
    obs = jnp.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[sorter]
    count = jnp.r_[jnp.argwhere(obs, size=size).T[0], obs.size]
    return .5 * (count[dense] + count[dense - 1] + 1)


def rank_fast_on_gpu(x):
    # See https://github.com/google/jax/issues/10434, jnp.argsort is slow on cpu.
    warnings.WarningMessage("jax.numpy.argsort is slower than np.argsort on cpu, considering using smx.math.rank_fast_on_cpu (namely scipy.stats.rankdata) instead")
    x = jnp.ravel(jnp.asarray(x))
    out_size = jnp.unique(x).size
    return _rank_dense_avg(x, out_size)

def rank_fast_on_cpu(x):
    return rank_scipy(x)


def rank(x):
    if jax.default_backend() == 'cpu':
        return rank_fast_on_cpu(x)
    else:
        return rank_fast_on_gpu(x)

