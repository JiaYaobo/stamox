"""Tests for one-way ANOVA."""
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.anova import one_way


class OneWayAOVTest(jtest.JaxTestCase):

    def test_one_way_1darray(self):
        tillamook = jnp.array([0.0571, 0.0813, 0.0831, 0.0976, 0.0817,
                               0.0859, 0.0735, 0.0659, 0.0923, 0.0836])
        newport = jnp.array([0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835,
                             0.0725])
        petersburg = jnp.array(
            [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105])
        magadan = jnp.array([0.1033, 0.0915, 0.0781, 0.0685,
                             0.0677, 0.0697, 0.0764, 0.0689])
        tvarminne = jnp.array(
            [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045])

        state = one_way(tillamook, newport, petersburg, magadan, tvarminne)
        self.assertAllClose(state.statistic, np.array(7.121019471642447))
        self.assertAllClose(state.p_value, np.array(0.0002812242314534544))

    def test_one_way_ndarray(self):
        a = np.array([[9.87, 9.03, 6.81],
                      [7.18, 8.35, 7.00],
                      [8.39, 7.58, 7.68],
                      [7.45, 6.33, 9.35],
                      [6.41, 7.10, 9.33],
                      [8.00, 8.24, 8.44]])

        b = np.array([[6.35, 7.30, 7.16],
                      [6.65, 6.68, 7.63],
                      [5.72, 7.73, 6.72],
                      [7.01, 9.19, 7.41],
                      [7.75, 7.87, 8.30],
                      [6.90, 7.97, 6.97]])

        c = np.array([[3.31, 8.77, 1.01],
                      [8.25, 3.24, 3.62],
                      [6.32, 8.81, 5.19],
                      [7.48, 8.83, 8.91],
                      [8.59, 6.01, 6.07],
                      [3.07, 9.72, 7.48]])

        state = one_way(a, b, c)
        self.assertAllClose(state.statistic, np.array(
            [1.75676344, 0.03701228, 3.76439349]))
        self.assertAllClose(state.p_value, np.array(
            [0.20630784, 0.96375203, 0.04733157]))


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
