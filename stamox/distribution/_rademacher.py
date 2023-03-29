from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_jit, filter_vmap
from jax import lax
from jax._src.random import Shape
from jax.random import KeyArray
from jaxtyping import ArrayLike, Float, Int

from ..core import make_partial_pipe


@filter_jit
def _prademacher(k: Union[Int, Float, ArrayLike]):
    cond0 = k < -1.0
    cond1 = jnp.logical_and(k >= -1.0, k < 1.0)
    cond2 = k >= 1.0
    index = jnp.argwhere(jnp.array([cond0, cond1, cond2]), size=1).squeeze()
    branches = [lambda: 0.0, lambda: 1 / 2, lambda: 1.0]
    p = lax.switch(index, branches, k)
    return p


@make_partial_pipe
def prademacher(
    k: Union[Int, Float, ArrayLike], lower_tail: bool = True, log_p: bool = False
):
    k = jnp.atleast_1d(k)
    p = filter_vmap(_prademacher)(k)
    if not lower_tail:
        p = 1 - p
    if log_p:
        p = jnp.log(p)
    return p


@filter_jit
def _drademacher(k: Union[Int, Float, ArrayLike]):
    in_support = jnp.logical_or(k == -1.0, k == 1.0)
    dens = jnp.where(in_support, 1.0 / 2.0, 0)
    return dens


@make_partial_pipe
def drademacher(
    k: Union[Int, Float, ArrayLike], lower_tail: bool = True, log_prob: bool = False
):
    k = jnp.atleast_1d(k)
    grads = filter_vmap(_drademacher)(k)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _rrademacher(key: KeyArray, sample_shape: Optional[Shape] = None):
    return jrand.rademacher(key, shape=sample_shape)


@make_partial_pipe
def rrademacher(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    lower_tail: bool = True,
    log_prob: bool = False,
):
    probs = _rrademacher(key, sample_shape=sample_shape)
    if not lower_tail:
        probs = 1 - probs
    if log_prob:
        probs = jnp.log(probs)
    return probs
