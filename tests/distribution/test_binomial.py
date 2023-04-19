"""Test for Binomial Distribution."""
import jax.numpy as jnp
import jax.random as jrand
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import binom

import stamox.pipe_functions as PF
from stamox.distribution import dbinom, pbinom, qbinom, rbinom


class TestBinom(jtest.JaxTestCase):
    def test_rbinom(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        n = 20
        p = 0.5
        rbins = rbinom(key, sample_shape, n, p)
        avg = rbins.mean()
        var = rbins.var(ddof=1)
        self.assertAllClose(avg, n * p, atol=1e-2)
        self.assertAllClose(var, n * p * (1 - p), atol=1e-2)

    def test_pbinom(self):
        """Test pbinom."""
        x = jnp.array([0, 1, 2, 3, 4, 5])
        n = 5
        p = 0.5
        expected = binom.cdf(x, n, p)
        actual = pbinom(x, n, p)
        self.assertAllClose(actual, expected)

    def test_dbinom(self):
        """Test dbinom."""
        x = jnp.array([0, 1, 2, 3, 4, 5])
        n = 5
        p = 0.5
        expected = binom.pmf(x, n, p)
        actual = dbinom(x, n, p)
        self.assertAllClose(actual, expected)

    def test_qbinom(self):
        """Test qbinom."""
        p = jnp.array([0.1, 0.5, 0.9])
        n = 5
        prob = 0.5
        expected = binom.ppf(p, n, prob)
        actual = qbinom(p, n, prob)
        self.assertAllClose(actual, expected.astype(actual.dtype))

    def test_pipe_pbinom(self):
        """Test pipe pbinom."""
        x = jnp.array([0, 1, 2, 3, 4, 5])
        n = 5
        p = 0.5
        expected = binom.cdf(x, n, p)
        actual = PF.pbinom(size=n, prob=p)(x)
        self.assertAllClose(actual, expected)

    def test_pipe_dbinom(self):
        """Test pipe dbinom."""
        x = jnp.array([0, 1, 2, 3, 4, 5])
        n = 5
        p = 0.5
        expected = binom.pmf(x, n, p)
        actual = PF.dbinom(size=n, prob=p)(x)
        self.assertAllClose(actual, expected)

    def test_pipe_qbinom(self):
        """Test pipe qbinom."""
        p = jnp.array([0.1, 0.5, 0.9])
        n = 5
        prob = 0.5
        expected = binom.ppf(p, n, prob)
        actual = PF.qbinom(p=p, size=n)(prob)
        self.assertAllClose(actual, expected.astype(actual.dtype))

    def test_pipe_rbinom(self):
        """Test pipe rbinom."""
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        n = 20
        p = 0.5
        rbins = PF.rbinom(sample_shape=sample_shape, size=n, prob=p)(key)
        avg = rbins.mean()
        var = rbins.var(ddof=1)
        self.assertAllClose(avg, n * p, atol=1e-2)
        self.assertAllClose(var, n * p * (1 - p), atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
