"""Test for laplace distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import laplace

import stamox.pipe_functions as PF
from stamox.distribution import dlaplace, plaplace, qlaplace, rlaplace


def test_rlaplace():
    key = jrand.PRNGKey(19751002)
    sample_shape = (100000000,)
    laplaces = rlaplace(key, sample_shape=sample_shape)
    mean = laplaces.mean()
    var = laplaces.var(ddof=1)
    np.testing.assert_allclose(mean, 0.0, atol=1e-2, rtol=1e-4)
    np.testing.assert_allclose(var, 2 * 1.0**2, atol=1e-2, rtol=1e-4)


def test_plaplace():
    x = np.random.laplace(0.0, 1.0, 1000)
    loc = 0.0
    scale = 1.0
    p = plaplace(x, loc, scale, dtype=np.float64)
    true_p = laplace(loc=loc, scale=scale).cdf(x)
    np.testing.assert_allclose(p, true_p)


def test_qlaplace():
    q = np.array([0.5475813, 0.5906346, 0.6295909, 0.6648400, 0.6967347])
    loc = 0.0
    scale = 1.0
    x = qlaplace(q, loc, scale)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_dlaplace():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    loc = 0.0
    scale = 1.0
    grads = dlaplace(x, loc, scale)
    true_grads = np.array([0.4524187, 0.4093654, 0.3704091, 0.3351600, 0.3032653])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_plaplace():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    loc = 0.0
    scale = 1.0
    p = PF.plaplace(loc=loc, scale=scale)(x)
    true_p = np.array([0.5475813, 0.5906346, 0.6295909, 0.6648400, 0.6967347])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qlaplace():
    q = np.array([0.5475813, 0.5906346, 0.6295909, 0.6648400, 0.6967347])
    loc = 0.0
    scale = 1.0
    x = PF.qlaplace(loc=loc, scale=scale)(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dlaplace():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    loc = 0.0
    scale = 1.0
    grads = PF.dlaplace(loc=loc, scale=scale)(x)
    true_grads = np.array([0.4524187, 0.4093654, 0.3704091, 0.3351600, 0.3032653])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rlaplace():
    key = jrand.PRNGKey(19751002)
    sample_shape = (100000000,)
    laplaces = PF.rlaplace(sample_shape=sample_shape)(key)
    mean = laplaces.mean()
    var = laplaces.var(ddof=1)
    np.testing.assert_allclose(mean, 0.0, atol=1e-2, rtol=1e-4)
    np.testing.assert_allclose(var, 2 * 1.0**2, atol=1e-2, rtol=1e-4)
