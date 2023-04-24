"""Test for beta distribution"""
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from scipy.stats import beta

import stamox.pipe_functions as PF
from stamox.distribution import dbeta, pbeta, qbeta, rbeta


key = jrand.PRNGKey(19751002)


def test_rbeta():
    sample_shape = (10000,)
    a = 2.0
    b = 2.0
    ts = rbeta(key, sample_shape, a, b)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, a / (a + b), atol=1e-2)
    np.testing.assert_allclose(var, a * b / (((a + b) ** 2) * (a + b + 1)), atol=1e-2)


def test_pbeta():
    x = jrand.uniform(key, (1000,))
    a = 2.0
    b = 2.0
    p = pbeta(x, a, b)
    true_p = beta.cdf(x, a, b)
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_qbeta():
    q = jrand.uniform(key, (1000,))
    x = qbeta(q, 2.0, 2.0)
    true_x = beta.ppf(q, 2.0, 2.0)
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_dbeta():
    x = jrand.uniform(key, (1000,))
    grads = dbeta(x, 2, 2)
    true_grads = beta.pdf(x, 2, 2)
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_pbeta():
    x = jnp.array([0.0, 1.0, 0.2, 0.3, 0.4, 0.5])
    a = 2.0
    b = 2.0
    p = PF.pbeta(a=a, b=b, lower_tail=False)(x)
    true_p = jnp.array([1.000, 0.000, 0.896, 0.784, 0.648, 0.500])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qbeta():
    q = jnp.array([0.000, 1.000, 0.104, 0.216, 0.352, 0.500])
    x = PF.qbeta(a=2.0, b=2.0, lower_tail=False)(q)
    true_x = jnp.array([1.0, 0.0, 0.8, 0.7, 0.6, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dbeta():
    x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    grads = PF.dbeta(a=2, b=2)(x)
    true_grads = jnp.array([0.54, 0.96, 1.26, 1.44, 1.50])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rbeta():
    sample_shape = (10000,)
    a = 2.0
    b = 2.0
    ts = PF.rbeta(a=a, b=b, sample_shape=sample_shape)(key)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, a / (a + b), atol=1e-2)
    np.testing.assert_allclose(var, a * b / (((a + b) ** 2) * (a + b + 1)), atol=1e-2)
