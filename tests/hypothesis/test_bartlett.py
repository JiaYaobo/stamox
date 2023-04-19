"""Test for Bartlett test"""
import numpy as np
from absl.testing import absltest
from equinox import filter_jit
from jax._src import test_util as jtest

import stamox.pipe_functions as PF
from stamox.core import Pipeable
from stamox.hypothesis import bartlett_test


class BartlettTest(jtest.JaxTestCase):
    def test_bartlett(self):
        a = np.array(
            [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99],
            dtype=np.float32,
        )
        b = np.array(
            [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05],
            dtype=np.float32,
        )
        c = np.array(
            [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98],
            dtype=np.float32,
        )
        state = bartlett_test(a, b, c)
        self.assertAllClose(state.p_value, np.array(1.1254782518834628e-05))

    def test_pipe_bartlett(self):
        a = np.array(
            [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99],
            dtype=np.float32,
        )
        b = np.array(
            [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05],
            dtype=np.float32,
        )
        c = np.array(
            [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98],
            dtype=np.float32,
        )
        h = Pipeable([a, b, c]) >> PF.bartlett_test
        state = h()
        self.assertAllClose(state.p_value, np.array(1.1254782518834628e-05))
    
    def test_pipe_bartlett_jit(self):
        a = np.array(
            [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99],
            dtype=np.float32,
        )
        b = np.array(
            [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05],
            dtype=np.float32,
        )
        c = np.array(
            [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98],
            dtype=np.float32,
        )
        h = Pipeable([a, b, c]) >> filter_jit(PF.bartlett_test)
        state = h()
        self.assertAllClose(state.p_value, np.array(1.1254782518834628e-05))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
