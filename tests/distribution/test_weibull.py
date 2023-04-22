"""Test for weibull distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import weibull_min

import stamox.pipe_functions as PF
from stamox.distribution import dweibull, pweibull, qweibull, rweibull


np.random.seed(1)


def test_rweibull():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    a = 1.0
    b = 1.0
    ts = rweibull(key, sample_shape, a, b)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, 1.0, atol=1e-2)
    np.testing.assert_allclose(var, 1.0, atol=1e-2)


def test_pweibull():
    x = np.random.uniform(0, 1, 10000)
    a = 1.0
    b = 1.0
    p = pweibull(x, a, b, dtype=np.float64)
    true_p = weibull_min.cdf(x, a, scale=b)
    np.testing.assert_allclose(p, true_p)


def test_qweibull():
    q = np.random.uniform(0, 0.9999, 10000)
    x = qweibull(q, 1.0, 1.0, dtype=np.float64)
    true_x = weibull_min.ppf(q, 1.0, scale=1.0)
    np.testing.assert_allclose(x, true_x, atol=1e-4)


def test_dweibull():
    x = np.random.weibull(1.0, 10000)
    grads = dweibull(x, 1.0, 1.0, dtype=np.float64)
    true_grads = weibull_min.pdf(x, 1.0, scale=1.0)
    np.testing.assert_allclose(grads, true_grads)


def test_partial_pweibull():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    a = 1.0
    b = 1.0
    p = PF.pweibull(concentration=a, scale=b)(x)
    true_p = np.array([0.09516258, 0.18126925, 0.25918178, 0.32967995, 0.39346934])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qweibull():
    q = np.array([0.09516258, 0.18126925, 0.25918178, 0.32967995, 0.39346934])
    a = 1.0
    b = 1.0
    x = PF.qweibull(concentration=a, scale=b)(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dweibull():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    a = 1.0
    b = 1.0
    grads = PF.dweibull(concentration=a, scale=b)(x)
    true_grads = np.array([0.9048374, 0.8187308, 0.7408182, 0.6703200, 0.6065307])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rweibull():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    a = 1.0
    b = 1.0
    rvs = PF.rweibull(concentration=a, scale=b, sample_shape=sample_shape)(key)
    avg = rvs.mean()
    var = rvs.var(ddof=1)
    np.testing.assert_allclose(avg, 1.0, atol=1e-2)
    np.testing.assert_allclose(var, 1.0, atol=1e-2)
