from jax import lax
from jax.scipy.special import betainc
from tensorflow_probability.substrates.jax.math import special as tfp_special


def fdtrc(a, b, x):
    w = lax.div(b, lax.add(b, lax.mul(a, x)))
    a = lax.mul(0.5, a)
    b = lax.mul(0.5, b)
    return betainc(a, b, w)


def fdtr(a, b, x):
    w = lax.mul(a, x)
    w = lax.div(w, lax.add(b, w))
    a = lax.mul(0.5, a)
    b = lax.mul(0.5, b)
    return betainc(a, b, w)


def fdtri(a, b, y):
    y = lax.sub(1.0, y)
    a = lax.mul(0.5, a)
    b = lax.mul(0.5, b)
    w = betainc(a, b, 0.5)
    cond0 = (w > y) | (y < 0.001)
    w = lax.select(
        cond0,
        tfp_special.betaincinv(b, a, y),
        tfp_special.betaincinv(a, b, lax.sub(1.0, y)),
    )
    left_out = lax.div(lax.sub(b, lax.mul(b, w)), lax.mul(a, w))
    right_out = lax.div(lax.mul(b, w), lax.mul(a, lax.sub(1.0, w)))
    x = lax.select(cond0, left_out, right_out)
    return x
