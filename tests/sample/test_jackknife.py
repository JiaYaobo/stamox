"""Test for Bootstrap Sampler."""
import jax.random as jrandom
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.core import Pipeable
from stamox.sample import jackknife_sample


class JackknifeSamplerTest(jtest.JaxTestCase):
    
    def test_jackknife_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        S = jackknife_sample(X)
        self.assertEqual(S.shape, (1000, 999, 3))
    
    def test_partial_boostrap_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        h = Pipeable(X) >> jackknife_sample
        S = h()
        self.assertEqual(S.shape, (1000, 999, 3))

if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())