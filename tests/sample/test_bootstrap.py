"""Test for Bootstrap Sampler."""
import jax.random as jrandom
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.core import Pipeable
from stamox.sample import bootstrap_sample


class BootstrapSamplerTest(jtest.JaxTestCase):
    
    def test_bootstrap_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        S = bootstrap_sample(X, 5, key=key)
        self.assertEqual(S.shape, (5, 1000, 3))
    
    def test_partial_boostrap_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        h = Pipeable(X) >> bootstrap_sample(num_samples=5, key=key)
        S = h()
        self.assertEqual(S.shape, (5, 1000, 3))



if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())