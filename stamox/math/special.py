import jax.numpy as jnp
from jax.scipy.special import betainc
from tensorflow_probability.substrates.jax.math import special as tfp_special


def fdtrc(a, b, x):
    w = b / (b + a * x)
    return betainc(0.5 * b, 0.5 * a, w)


def fdtr(a, b, x):
    w = a * x
    w = w / (b + w)
    return betainc(0.5 * a, 0.5 * b, w)


def fdtri(a, b, y):
    y = 1.0 - y
    w = betainc(0.5 * b, 0.5 * a, 0.5)
    cond0 = (w > y) | (y < 0.001)
    w = jnp.where(
        cond0,
        tfp_special.betaincinv(0.5 * b, 0.5 * a, y),
        tfp_special.betaincinv(0.5 * a, 0.5 * b, 1.0 - y),
    )
    x = jnp.where(cond0, (b - b * w) / (a * w), b * w / (a * (1.0 - w)))
    return x
