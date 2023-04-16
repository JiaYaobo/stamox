"""Pearson correlation coefficient and p-value"""

import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import ArrayLike

from ...distribution import pbeta


def pearsonr(
    x: ArrayLike,
    y: ArrayLike,
    formula: str = None,
    dtype=jnp.float32,
    *,
    alternative="two-sided"
):
    x = jnp.asarray(x, dtype=dtype)
    y = jnp.asarray(y, dtype=dtype)
    # check ndim and shape
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    # check formula
    if formula is not None:
        raise NotImplementedError("formula is not implemented yet")
    # check alternative
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    # compute
    return _pearsonr(x, y, dtype=dtype, alternative=alternative)


@filter_jit
def _pearsonr(
    x: ArrayLike, y: ArrayLike, dtype=jnp.float32, *, alternative="two-sided"
):
    """Pearson correlation coefficient and p-value"""
    n = x.shape[0]
    x = jnp.asarray(x, dtype=dtype)
    y = jnp.asarray(y, dtype=dtype)
    x_centered = x - jnp.mean(x)
    x_std = x_centered / jnp.linalg.norm(x_centered)
    y_centered = y - jnp.mean(y)
    y_std = y_centered / jnp.linalg.norm(y_centered)
    r = jnp.dot(x_std, y_std)
    r = jnp.maximum(jnp.minimum(r, 1.0), -1.0)
    ab = n / 2 - 1
    scaled_r = (1 - r) / 2
    if alternative == "two-sided":
        p = 2 * pbeta(jnp.abs(scaled_r), ab, ab)
    elif alternative == "less":
        p = 1 - pbeta(scaled_r, ab, ab)
    elif alternative == "greater":
        p = pbeta(scaled_r, ab, ab)
    return r, p
