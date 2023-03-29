"""Test for Durbin-Watson test"""
import numpy as np
from absl.testing import absltest
from equinox import filter_jit
from jax._src import test_util as jtest
from statsmodels.stats.stattools import durbin_watson

from stamox.core import Pipeable
from stamox.hypothesis import durbin_watson_test


class DurbinWastonTest(jtest.JaxTestCase):
    def test_durbin_waston(self):
        x = np.array([0.1, 0.2, 0.3] * 50, dtype=np.float32)
        state = durbin_watson_test(x)
        self.assertAllClose(state.statistic, np.array([durbin_watson(x)]), atol=1e-3)
        
    def test_pipe_durbin_waston(self):
        x = np.array([0.1, 0.2, 0.3] * 50, dtype=np.float32)
        h = Pipeable(x) >> durbin_watson_test
        state = h()
        self.assertAllClose(state.statistic, np.array([durbin_watson(x)]), atol=1e-3)
    
    def test_pipe_durbin_waston_jit(self):
        x = np.array([0.1, 0.2, 0.3] * 50, dtype=np.float32)
        h = filter_jit(Pipeable(x) >> durbin_watson_test)
        state = h()
        self.assertAllClose(state.statistic, np.array([durbin_watson(x)]), atol=1e-3)

if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())