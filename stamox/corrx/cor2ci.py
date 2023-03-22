import jax.numpy as jnp

from ..distribution import qnorm


def cor2ci_kendall(cor, n, ci=0.95, correction="fieller"):
    if correction == "fieller":
        tau_se = (0.437 / (n - 4)) ** 0.5
    else:
        tau_se = 1 / (n - 3) ** 0.5

    moe = qnorm(1 - (1 - ci) / 2) * tau_se

    zu = jnp.arctanh(cor) + moe
    zl = jnp.arctanh(cor) - moe

    ci_low = jnp.tanh(zl)
    ci_high = jnp.tanh(zu)

    return ci_low, ci_high


def cor2ci_spearman(cor, n, ci=0.95, correction="fieller"):
    if correction == "fieller":
        zrs_se = (1.06 / (n - 3)) ** 0.5
    elif correction == "bw":
        zrs_se = ((1 + cor ** 2 / 2) / (n - 3)) ** 0.5
    else:
        zrs_se = 1 / (n - 3) ** 0.5

    moe = qnorm(1 - (1 - ci) / 2) * zrs_se

    zu = jnp.arctanh(cor) + moe
    zl = jnp.arctanh(cor) - moe

    ci_low = jnp.tanh(zl)
    ci_high = jnp.tanh(zu)

    return ci_low, ci_high


def cor2ci_pearson(cor, n, ci=0.95, *args):
    z = jnp.arctanh(cor)
    se = 1 / jnp.sqrt(n - 3)

    alpha = 1 - (1 - ci) / 2

    ci_low = z - se * qnorm(alpha)
    ci_high = z + se * qnorm(alpha)

    ci_low = jnp.tanh(ci_low)
    ci_high = jnp.tanh(ci_high)

    return ci_low, ci_high
