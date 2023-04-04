"""Test for Bootstrap Sampler."""
import jax.random as jrandom
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.basic import mean
from stamox.core import Pipeable
from stamox.sample import jackknife, jackknife_sample


class JackknifeSamplerTest(jtest.JaxTestCase):
    def test_jackknife_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        S = jackknife_sample(X)
        self.assertEqual(S.shape, (1000, 999, 3))

    def test_partial_pipe_jackknife_sampler(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        h = Pipeable(X) >> jackknife_sample
        S = h()
        self.assertEqual(S.shape, (1000, 999, 3))

    def test_jackknife(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        S = jackknife(X, mean)
        self.assertAllClose(mean(S), mean(X), atol=1e-2)

    def test_partial_pipe(self):
        key = jrandom.PRNGKey(20010813)
        X = jrandom.normal(key=key, shape=(1000, 3))
        h = Pipeable(X) >> jackknife(call=mean)
        S = h()
        self.assertAllClose(mean(S), mean(X), atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
