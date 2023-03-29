import jax.numpy as jnp
from equinox import filter_jit
from jax import lax, vmap

from ..core import make_pipe
from ..distribution import pnorm, qnorm
from ._base import HypoTest


_g = [-2.273, 0.459]
_c1 = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056]
_c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633]
_c3 = [0.544, -0.39978, 0.025054, -6.714e-4]
_c4 = [1.3822, -0.77857, 0.062767, -0.0020322]
_c5 = [-1.5861, -0.31082, -0.083751, 0.0038915]
_c6 = [-0.4803, -0.082676, 0.0030302]
_a_1 = 0.70710678  # sqrt(2)


class ShapiroWilkTest(HypoTest):
    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
        name="Shapiro-Wilk Test",
    ):
        super().__init__(
            statistic, parameters, p_value, estimate, null_value, alternative, name
        )


@make_pipe
def shapiro_wilk_test(x):
    """Computes the Shapiro-Wilk test for normality.

    Args:
        x (jnp.ndarray): The data to be tested.

    Returns:
        w (float): The Shapiro-Wilk statistic.
    """
    x = jnp.ravel(x)
    x = jnp.sort(x)
    n = x.size
    x = x - jnp.median(x)
    w = _shapiro_wilk(x, n)
    return w


@filter_jit
def _shapiro_wilk(x, n):
    nn2 = n // 2
    an = jnp.asarray(n, dtype=jnp.float32)
    if n < 3:
        raise ValueError("Data must be at least length 3.")

    an25 = an + 0.25
    a = vmap(lambda i: qnorm(jnp.asarray((i - 0.375) / an25), 0.0, 1.0), in_axes=0)(
        jnp.arange(nn2, dtype=jnp.float32) + 1
    )
    a = jnp.squeeze(a, axis=-1)
    summ2 = 2 * jnp.sum(a**2, axis=0)
    ssumm2 = jnp.sqrt(summ2)
    rsn = 1.0 / jnp.sqrt(an)
    a1 = poly(rsn, _c1, 6) - a[0] / ssumm2

    # Normalize
    if n > 5:
        a2 = -a[1] / ssumm2 + poly(rsn, _c2, 6)
        fac = jnp.sqrt(
            (summ2 - 2.0 * (a[0] * a[0]) - 2.0 * (a[1] * a[1]))
            / (1.0 - 2.0 * (a1 * a1) - 2.0 * (a2 * a2))
        )
    else:
        fac = jnp.sqrt((summ2 - 2.0 * (a[0] * a[0])) / (1.0 - 2.0 * (a1 * a1)))
    if n > 3:
        a2 = a[1] / -fac
        aa = jnp.r_[a1, a2, jnp.zeros((n // 2 - 2,))]
        aaa = jnp.r_[0.0, 0.0, a[2:]] / -fac
        a = aaa + aa
    else:
        a = jnp.r_[a1]

    range_x = x[n - 1] - x[0]
    sx = jnp.sum(x, axis=0) / range_x

    forward = jnp.arange(1, n, 1, dtype=jnp.float32)
    reverse = jnp.arange(n - 1, 0, -1, dtype=jnp.float32) - 1
    sa = jnp.sum(
        vmap(
            lambda i, j: jnp.sign(i - j) * a[jnp.int32(jnp.minimum(i, j))],
            in_axes=(0, 0),
        )(forward, reverse),
        axis=0,
    )
    sa += -a[0]
    sa /= n
    sx /= n

    def _tmp(i, j, xi):
        asa = jnp.sign(i - j) * a[jnp.int32(jnp.minimum(i, j))] - sa
        xsx = xi / range_x - sx
        ssa = asa * asa
        ssx = xsx * xsx
        sax = asa * xsx
        return (ssa, ssx, sax)

    forward = jnp.arange(0, n, 1, dtype=jnp.float32)
    reverse = jnp.arange(n, 0, -1, dtype=jnp.float32) - 1
    (ssa, ssx, sax) = vmap(_tmp, in_axes=(0, 0, 0))(forward, reverse, x)
    ssa = jnp.sum(ssa, axis=0, keepdims=True)
    ssx = jnp.sum(ssx, axis=0, keepdims=True)
    sax = jnp.sum(sax, axis=0, keepdims=True)

    ssassx = jnp.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    w = 1.0 - w1

    # cal pw
    if n == 3:
        pi6 = 1.90985931710274
        stqr = 1.04719755119660
        pw = pi6 * (jnp.arcsin(jnp.sqrt(w)) - stqr)
        return ShapiroWilkTest(statistic=w, p_value=pw)

    y = jnp.log(w1)
    xx = jnp.log(an)
    if n <= 11:
        gamma = poly(an, _g, 2)

        pw = lax.select(
            y >= gamma,
            jnp.array([0.0]),
            pnorm(
                -jnp.log(gamma - y),
                poly(an, _c3, 4),
                jnp.exp(poly(an, _c4, 4)),
                lower_tail=False,
            ),
        )
    else:
        m = poly(xx, _c5, 4)
        s = jnp.exp(poly(xx, _c6, 3))
        pw = pnorm(y, m, s, lower_tail=False)
    return ShapiroWilkTest(statistic=w, p_value=pw)


@filter_jit
def poly(x, coef, norder):
    coef = jnp.asarray(coef)
    iss = jnp.arange(norder, dtype=jnp.float32)
    value = vmap(lambda c, i, x: x**i * c, in_axes=(0, 0, None))(coef, iss, x)
    return jnp.sum(value, axis=0, keepdims=True)
