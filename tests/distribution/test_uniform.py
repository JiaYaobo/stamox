"""Test the uniform distribution."""
import jax.random as jrand
import numpy as np
from scipy.stats import uniform

import stamox.pipe_functions as PF
from stamox.distribution import dunif, punif, qunif, runif


def test_runif():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    ts = runif(key, sample_shape)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, 1 / 2, atol=1e-2)
    np.testing.assert_allclose(var, 1 / 12, atol=1e-2)


def test_punif():
    x = np.random.uniform(0, 1, 10000)
    p = punif(x)
    true_p = uniform.cdf(x)
    np.testing.assert_allclose(p, true_p)


def test_qunif():
    q = np.random.uniform(0, 1, 10000)
    x = qunif(q)
    true_x = uniform.ppf(q)
    np.testing.assert_allclose(x, true_x)


def test_dunif():
    x = np.random.uniform(0, 1, 10000)
    grads = dunif(x)
    true_grads = np.ones_like(x)
    np.testing.assert_allclose(grads, true_grads)


def test_partial_punif():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p = PF.punif()(x)
    true_p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(p, true_p)


def test_partial_qunif():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    x = PF.qunif()(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x)


def test_partial_dunif():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    grads = PF.dunif()(x)
    true_grads = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(grads, true_grads)


def test_partial_runif():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    rvs = PF.runif()(key, sample_shape)
    avg = rvs.mean()
    var = rvs.var(ddof=1)
    np.testing.assert_allclose(avg, 1 / 2, atol=1e-2)
    np.testing.assert_allclose(var, 1 / 12, atol=1e-2)
