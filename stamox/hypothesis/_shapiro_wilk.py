import functools as ft

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from stamox.distribution import qnorm

_g = [-2.273, .459]
_c1 = [0., .221157, -.147981, -2.07119, 4.434685, -2.706056]
_c2 = [0., .042981, -.293762, -1.752461, 5.682633, -3.582633]
_c3 = [.544, -.39978, .025054, -6.714e-4]
_c4 = [1.3822, -.77857, .062767, -.0020322]
_c5 = [-1.5861, -.31082, -.083751, .0038915]
_c6 = [-.4803, -.082676, .0030302]
_a_1 = 0.70710678  # sqrt(2)


def shapiro_wilk(x):
    x = jnp.ravel(x)
    # since numpy sort algorithms is much faster than jax.numpy, we use numpy to do sort things
    # on gpu jnp.sort will be fatser though
    if jax.default_backend() == "cpu":
        x = np.sort(x)
    else:
        x = jnp.sort(x)
    n = x.size
    x = x - jnp.median(x)
    w = _shapiro_wilk(x, n)
    return w


@ft.partial(jit, static_argnames=('n', ))
def _shapiro_wilk(x, n):
    nn2 = n // 2
    an = jnp.asarray(n, dtype=jnp.float32)
    if n < 3:
        raise ValueError("Data must be at least length 3.")

    an25 = n + .25
    a = vmap(lambda i: qnorm(jnp.asarray((i - 0.375) / an25), 0., 1.),
             in_axes=0)(jnp.arange(nn2) + 1)
    a = jnp.squeeze(a, axis=-1)
    summ2 = 2 * jnp.sum(a**2, axis=0)
    ssumm2 = jnp.sqrt(summ2)
    rsn = 1. / jnp.sqrt(an)
    a1 = poly(rsn, _c1, 6) - a[0] / ssumm2

    # Normalize
    if n > 5:
        a2 = -a[1] / ssumm2 + poly(rsn, _c2, 6)
        fac = jnp.sqrt((summ2 - 2. * (a[0] * a[0]) - 2. * (a[1] * a[1]))
                       / (1. - 2. * (a1 * a1) - 2. * (a2 * a2)))
    else:
        fac = jnp.sqrt((summ2 - 2. * (a[0] * a[0]))
                       / (1. - 2. * (a1 * a1)))
    if n > 3:
        a2 = a[1] / -fac
        aa = jnp.r_[a1, a2, jnp.zeros((n // 2 - 2, ))]
        aaa = jnp.r_[0., 0., a[2:]] / -fac
        a = aaa + aa
    else:
        a = jnp.r_[a1]

    range_x = x[n - 1] - x[0]
    sx = jnp.sum(x, axis=0) / range_x

    forward = jnp.arange(1, n, 1)
    reverse = jnp.arange(n - 1, 0, -1) - 1
    sa = jnp.sum(vmap(lambda i, j: jnp.sign(i - j) *
                 a[jnp.minimum(i, j)], in_axes=(0, 0))(forward, reverse), axis=0)
    sa += -a[0]
    sa /= n
    sx /= n

    def _tmp(i, j, xi):
        asa = jnp.sign(i - j) * a[jnp.minimum(i, j)] - sa
        xsx = xi / range_x - sx
        ssa = asa * asa
        ssx = xsx * xsx
        sax = asa * xsx
        return (ssa, ssx, sax)

    forward = jnp.arange(0, n, 1)
    reverse = jnp.arange(n, 0, -1) - 1
    (ssa, ssx, sax) = vmap(_tmp, in_axes=(0, 0, 0))(forward, reverse, x)
    ssa = jnp.sum(ssa, axis=0)
    ssx = jnp.sum(ssx, axis=0)
    sax = jnp.sum(sax, axis=0)

    ssassx = jnp.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    w = 1. - w1

    # cal pw
    

    return w


@ft.partial(jit, static_argnames=('norder',))
def poly(x, coef, norder):
    coef = jnp.asarray(coef)
    iss = jnp.arange(norder)
    value = vmap(lambda c, i, x: x ** i * c,
                 in_axes=(0, 0, None))(coef, iss, x)
    return jnp.sum(value, axis=0)
