"""Test for Binomial Distribution."""
import jax.random as jrand
import numpy as np
from scipy.stats import binom

import stamox.pipe_functions as PF
from stamox.distribution import dbinom, pbinom, qbinom, rbinom


def test_rbinom():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    n = 20
    p = 0.5
    rbins = rbinom(key, sample_shape, n, p)
    avg = rbins.mean()
    var = rbins.var(ddof=1)
    np.testing.assert_allclose(avg, n * p, atol=1e-2)
    np.testing.assert_allclose(var, n * p * (1 - p), atol=1e-2)


def test_pbinom():
    """Test pbinom."""
    x = np.array([0, 1, 2, 3, 4, 5])
    n = 5
    p = 0.5
    expected = binom.cdf(x, n, p)
    actual = pbinom(x, n, p)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_dbinom():
    """Test dbinom."""
    x = np.array([0, 1, 2, 3, 4, 5])
    n = 5
    p = 0.5
    expected = binom.pmf(x, n, p)
    actual = dbinom(x, n, p)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_qbinom():
    """Test qbinom."""
    p = np.array([0.1, 0.5, 0.9])
    n = 5
    prob = 0.5
    expected = binom.ppf(p, n, prob)
    actual = qbinom(p, n, prob)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_pipe_pbinom():
    """Test pipe pbinom."""
    x = np.array([0, 1, 2, 3, 4, 5])
    n = 5
    p = 0.5
    expected = binom.cdf(x, n, p)
    actual = PF.pbinom(size=n, prob=p)(x)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_pipe_dbinom():
    """Test pipe dbinom."""
    x = np.array([0, 1, 2, 3, 4, 5])
    n = 5
    p = 0.5
    expected = binom.pmf(x, n, p)
    actual = PF.dbinom(size=n, prob=p)(x)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_pipe_qbinom():
    """Test pipe qbinom."""
    p = np.array([0.1, 0.5, 0.9])
    n = 5
    prob = 0.5
    expected = binom.ppf(p, n, prob)
    actual = PF.qbinom(prob=prob, size=n)(p)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_pipe_rbinom():
    """Test pipe rbinom."""
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    n = 20
    p = 0.5
    rbins = PF.rbinom(sample_shape=sample_shape, size=n, prob=p)(key)
    avg = rbins.mean()
    var = rbins.var(ddof=1)
    np.testing.assert_allclose(avg, n * p, atol=1e-2)
    np.testing.assert_allclose(var, n * p * (1 - p), atol=1e-2)
