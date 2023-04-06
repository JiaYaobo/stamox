from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float

from ..core import make_partial_pipe


@filter_jit
def _pexp(x: Union[float, ArrayLike], rate: float) -> Float:
    return -jnp.expm1(-rate * x)


@make_partial_pipe
def pexp(
    q: Union[float, ArrayLike], rate: float, lower_tail=True, log_prob=False
) -> Float:
    p = filter_vmap(_pexp)(q, rate)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


@filter_jit
def _qexp(q: Union[float, ArrayLike], rate: float) -> Float:
    return -jnp.log1p(-q) / rate


@make_partial_pipe
def qexp(
    p: Union[float, ArrayLike],
    rate: Union[float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    x = filter_vmap(_qexp)(p, rate)
    return x


_dexp = filter_jit(filter_grad(_pexp))


@make_partial_pipe
def dexp(x: Union[float, ArrayLike], rate: float, lower_tail=True, log_prob=False):
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dexp)(x, rate)
    if not lower_tail:
        grads = -grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _rexp(
    key: KeyArray,
    rate: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.shape(rate)
    rate = jnp.broadcast_to(rate, sample_shape)
    return jrand.exponential(key, shape=sample_shape) / rate


@make_partial_pipe
def rexp(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    rate: Union[Float, ArrayLike] = None,
    lower_tail=True,
    log_prob=False,
):
    rvs = _rexp(key, rate, sample_shape)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
