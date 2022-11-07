import functools as ft

import jax.numpy as jnp
from jax import jit, vmap


from stamox.distrix import qnorm

_g = [-2.273, .459]
_c1 = [0.,.221157,-.147981,-2.07119, 4.434685, -2.706056]
_c2 = [0.,.042981,-.293762,-1.752461,5.682633, -3.582633]
_c3 = [.544,-.39978,.025054,-6.714e-4]
_c4 = [1.3822,-.77857,.062767,-.0020322]
_c5 = [-1.5861,-.31082,-.083751,.0038915]
_c6 = [-.4803,-.082676,.0030302]
_a_1 = 0.70710678 # sqrt(2)


def _shapiro_wilk(x, n):
    x = jnp.ravel(x)
    nn2 = n / 2
    an = jnp.asarray(n, dtype=jnp.float32)
    if n < 3:
        raise ValueError("Data must be at least length 3.")

    x = x - jnp.median(x)

    an25 = n + .25
    a = vmap(lambda i: qnorm((i - 0.375) / an25, 0., 1.))(jnp.arange(nn2) + 1)
    summ2 = jnp.sum(a**2, axis=0)
    ssumm2 = jnp.sqrt(summ2)
    rsn = 1. / jnp.sqrt(an)
    a1 = poly(_c1, 6, rsn) - a[0] / ssumm2

    # Normalize
    if n > 5:
        i1 = 3
        a2 = -a[1] / ssumm2 + poly(_c2, 6, rsn)
        fac = jnp.sqrt((summ2 - 2. * (a[0] * a[0]) - 2. * (a[1] * a[1])) \
                /(1. - 2. * (a1 * a1) - 2. * (a2 * a2)))
    else:
        i1 = 2
        fac = jnp.sqrt((summ2 - 2. * (a[0] * a[0])) \
                / (1. - 2. * (a1 * a1)))
    Range = x[n - 1] - x[0] 
    sx = jnp.sum(x, axis=0) / Range

    forward = jnp.arange(n)
    reverse = jnp.arange(n - 1, 0, -1)
    sa = jnp.sum(vmap(lambda i, j: jnp.sign(i - j) * a[jnp.minimum(i, j)])(forward, reverse), axis=0)
    sa /= n
    sx /= n

    def _tmp(i, j, xi):
        asa = jnp.sign(i - j) * a[jnp.minimum(i, j)] - sa
        xsx = xi / Range - sx
        ssa = asa * asa
        ssx = xsx * xsx
        sax = asa * xsx
        return (ssa, ssx, sax)
    
    (ssa, ssx, sax) = vmap(_tmp, in_axes=(0, 0, 0))(forward, reverse, x)
    ssa = jnp.sum(ssa, axis=0)
    ssx = jnp.sum(ssx, axis=0)
    sax = jnp.sum(sax, axis=0)

    ssassx = jnp.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    w = 1. - w1

    return w


@ft.partial(jit, static_argnames=('norder',))
def poly(x, coef, norder):
    coef = jnp.asarray(coef)
    iss = jnp.arange(norder)
    value = vmap(lambda c, i, x: x ** i * c, in_axes=(0, 0, None))(coef, iss, x)
    return jnp.sum(value, axis=0)
