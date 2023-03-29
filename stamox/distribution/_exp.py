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


@make_partial_pipe(name="pexp")
def pexp(
    x: Union[float, ArrayLike], rate: float, lower_tail=True, log_prob=False
) -> Float:
    p = filter_vmap(_pexp)(x, rate)
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
    q: Union[float, ArrayLike],
    rate: Union[float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    x = filter_vmap(_qexp)(q, rate)
    return x


_dexp = filter_jit(filter_grad(_pexp))


@make_partial_pipe(name="dexp")
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
    rate: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    lower_tail=True,
    log_prob=False,
):
    sample = _rexp(key, rate, sample_shape)
    if not lower_tail:
        sample = 1 - sample
    if log_prob:
        sample = jnp.log(sample)
    return sample
