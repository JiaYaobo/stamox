"""Test for Shapiro-Wilk test"""
import jax.random as jrandom
import numpy as np
from absl.testing import absltest
from equinox import filter_jit
from jax._src import test_util as jtest
from scipy.stats import shapiro

import stamox.pipe_functions as PF
from stamox.core import Pipeable
from stamox.hypothesis import shapiro_wilk_test


class ShapiroWilkTest(jtest.JaxTestCase):
    def test_shapiro_wilk(self):
        key = jrandom.PRNGKey(0)
        x = jrandom.normal(key, (100,))
        state = shapiro_wilk_test(x)
        osp_state = shapiro(x)
        self.assertAllClose(state.statistic, np.array(osp_state.statistic), atol=1e-3)
        self.assertAllClose(state.p_value, np.array(osp_state.pvalue), atol=1e-3)

    def test_pipe_shapiro_wilk(self):
        key = jrandom.PRNGKey(0)
        x = jrandom.normal(key, (100,))
        h = Pipeable(x) >> PF.shapiro_wilk_test
        state = h()
        osp_state = shapiro(x)
        self.assertAllClose(state.statistic, np.array(osp_state.statistic), atol=1e-3)
        self.assertAllClose(state.p_value, np.array(osp_state.pvalue), atol=1e-3)

    def test_pipe_shapiro_wilk_jit(self):
        key = jrandom.PRNGKey(0)
        x = jrandom.normal(key, (100,))
        h = filter_jit(Pipeable(x) >> PF.shapiro_wilk_test)
        state = h()
        osp_state = shapiro(x)
        self.assertAllClose(state.statistic, np.array(osp_state.statistic), atol=1e-3)
        self.assertAllClose(state.p_value, np.array(osp_state.pvalue), atol=1e-3)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
