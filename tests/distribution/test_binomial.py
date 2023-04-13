"""Test for Binomial Distribution."""
import jax.numpy as jnp
import jax.random as jrand
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import binom

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
    


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
