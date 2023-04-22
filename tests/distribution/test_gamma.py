"""Test for gamma distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import gamma

import stamox.pipe_functions as PF
from stamox.distribution import dgamma, pgamma, qgamma, rgamma


def test_rgamma():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    shape = 2.0
    rate = 2.0
    gammas = rgamma(key, sample_shape, shape, rate)
    avg = gammas.mean()
    var = gammas.var(ddof=1)
    np.testing.assert_allclose(avg, shape / rate, atol=1e-2)
    np.testing.assert_allclose(var, shape / rate**2, atol=1e-2)


def test_pgamma():
    x = np.random.gamma(2, 2, 1000)
    shape = 2.0
    rate = 2.0
    p = pgamma(x, shape, rate, dtype=np.float64)
    true_p = gamma.cdf(x, shape, scale=1 / rate)
    np.testing.assert_allclose(p, true_p)


def test_qgamma():
    q = np.random.uniform(0, 0.999, 1000)
    shape = 2.0
    rate = 2.0
    x = qgamma(q, shape, rate, dtype=np.float64)
    true_x = gamma.ppf(q, shape, scale=1 / rate)
    np.testing.assert_allclose(x, true_x)


def test_dgamma():
    x = np.random.gamma(2, 2, 1000)
    shape = 2.0
    rate = 2.0
    grads = dgamma(x, shape, rate, dtype=np.float64)
    true_grads = gamma.pdf(x, shape, scale=1 / rate)
    np.testing.assert_allclose(grads, true_grads)


def test_partial_pgamma():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    shape = 2.0
    rate = 2.0
    p = PF.pgamma(shape=shape, rate=rate)(x)
    true_p = np.array([0.01752310, 0.06155194, 0.12190138, 0.19120786, 0.26424112])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qgamma():
    q = np.array([0.01752310, 0.06155194, 0.12190138, 0.19120786, 0.26424112])
    shape = 2.0
    rate = 2.0
    x = PF.qgamma(shape=shape, rate=rate)(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dgamma():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    shape = 2.0
    rate = 2.0
    grads = PF.dgamma(shape=shape, rate=rate)(x)
    true_grads = np.array([0.3274923, 0.5362560, 0.6585740, 0.7189263, 0.7357589])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rgamma():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    shape = 2.0
    rate = 2.0
    gammas = PF.rgamma(shape=shape, rate=rate, sample_shape=sample_shape)(key)
    avg = gammas.mean()
    var = gammas.var(ddof=1)
    np.testing.assert_allclose(avg, shape / rate, atol=1e-2)
    np.testing.assert_allclose(var, shape / rate**2, atol=1e-2)
