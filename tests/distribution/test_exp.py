"""Test for exponential distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import expon

import stamox.pipe_functions as PF
from stamox.distribution import dexp, pexp, qexp, rexp


np.random.seed(1)


def test_rexp():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    rate = 2.5
    rvs = rexp(key, sample_shape, rate)
    np_rvs = np.random.exponential(1 / rate, sample_shape)
    np_avg = np_rvs.mean()
    np_var = np_rvs.var(ddof=1)
    avg = rvs.mean()
    var = rvs.var(ddof=1)
    np.testing.assert_allclose(avg, np_avg, atol=1e-2)
    np.testing.assert_allclose(var, np_var, atol=1e-2)


def test_pexp():
    x = np.random.exponential(1 / 2.5, 1000)
    rate = 2.5
    p = pexp(x, rate)
    true_p = expon(scale=1 / rate).cdf(x)
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_qexp():
    q = np.random.uniform(0, 0.999, 1000)
    x = qexp(q, 2.5, dtype=np.float64)
    true_x = expon(scale=1 / 2.5).ppf(q)
    np.testing.assert_allclose(x, true_x, atol=1e-15)


def test_dexp():
    x = np.random.exponential(1 / 2.5, 1000)
    rate = 2.5
    grads = dexp(x, rate)
    true_grads = np.exp(-x * rate) * rate
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_pexp():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rate = 2.5
    p = PF.pexp(rate=rate)(x)
    true_p = np.array([0.2211992, 0.3934693, 0.5276334, 0.6321206, 0.7134952])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qexp():
    q = np.array([0.2211992, 0.3934693, 0.5276334, 0.6321206, 0.7134952])
    x = PF.qexp(rate=2.5)(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dexp():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rate = 2.5
    grads = PF.dexp(rate=rate)(x)
    true_grads = np.array([1.9470020, 1.5163266, 1.1809164, 0.9196986, 0.7162620])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rexp():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    rate = 2.5
    rvs = PF.rexp(rate=rate, sample_shape=sample_shape)(key)
    avg = rvs.mean()
    var = rvs.var(ddof=1)
    np.testing.assert_allclose(avg, 1 / rate, atol=1e-2)
    np.testing.assert_allclose(var, 1 / rate**2, atol=1e-2)
