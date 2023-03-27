"""Test for Shapiro-Wilk test"""
from absl.testing import absltest

from jax._src import test_util as jtest
import  jax.random as jrandom
import numpy as np
from scipy.stats import shapiro
from equinox import filter_jit

from stamox.hypothesis import shapiro_wilk_test
from stamox.core import Pipeable

class ShapiroWilkTest(jtest.JaxTestCase):
    def test_shapiro_wilk(self):
        key = jrandom.PRNGKey(0)
        x = jrandom.normal(key, (100,))
        state = shapiro_wilk_test(x)
        osp_state = shapiro(x)
        self.assertAllClose(state.statistic, np.array([osp_state.statistic]), atol=1e-3)
        self.assertAllClose(state.p_value, np.array([osp_state.pvalue]), atol=1e-3)

    def test_pipe_shapiro_wilk(self):
        key = jrandom.PRNGKey(0)
        x = jrandom.normal(key, (100,))
        h = Pipeable(x) >> shapiro_wilk_test
        state = h()
        osp_state = shapiro(x)
        self.assertAllClose(state.statistic, np.array([osp_state.statistic]), atol=1e-3)
        self.assertAllClose(state.p_value, np.array([osp_state.pvalue]), atol=1e-3)

    def test_pipe_shapiro_wilk_jit(self):
        key = jrandom.PRNGKey(0)
        x = jrandom.normal(key, (100,))
        h = filter_jit(Pipeable(x) >> shapiro_wilk_test)
        state = h()
        osp_state = shapiro(x)
        self.assertAllClose(state.statistic, np.array([osp_state.statistic]), atol=1e-3)
        self.assertAllClose(state.p_value, np.array([osp_state.pvalue]), atol=1e-3)

if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())