"""Test for Bootstrap Sampler."""
import jax.random as jrandom
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.basic import mean
from stamox.core import Pipeable
from stamox.sample import bootstrap, bootstrap_sample


class BootstrapSamplerTest(jtest.JaxTestCase):
    
    def test_bootstrap_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        S = bootstrap_sample(X, 5, key=key)
        self.assertEqual(S.shape, (5, 1000, 3))
    
    def test_partial_pipe_boostrap_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        h = Pipeable(X) >> bootstrap_sample(num_samples=5, key=key)
        S = h()
        self.assertEqual(S.shape, (5, 1000, 3))
    
    def test_bootstrap(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        S = bootstrap(X, mean, 100, key=key)
        self.assertAllClose(mean(S), mean(X), atol=1e-2)
    

    def test_partial_pipe_bootstrap(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        h = Pipeable(X) >> bootstrap(call=mean, num_samples=100, key=key)
        S = h()
        self.assertAllClose(mean(S), mean(X), atol=1e-2)
    
    



if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())