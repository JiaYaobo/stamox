"""Test for beta distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest

from stamox.regression import ols


class OLSTest(jtest.JaxTestCase):

    def test_ols_pinv(self):
        key = jrand.PRNGKey(19751001)

        data_points, data_dimension = 100, 10

        # Generate X and w, then set y = Xw + ϵ
        X = jrand.normal(key, (data_points, data_dimension))

        true_w = jrand.normal(key, (data_dimension,))
        y = X.dot(true_w) + 0.001 * jrand.normal(key, (data_points,))
        w = ols(X, y, 'pinv')
        self.assertArraysAllClose(w, true_w, atol=1e-2)

    def test_ols_qr(self):
        key = jrand.PRNGKey(19751001)

        data_points, data_dimension = 100, 10
        X = jrand.normal(key, (data_points, data_dimension))

        true_w = jrand.normal(key, (data_dimension,))
        y = X.dot(true_w) + 0.001 * jrand.normal(key, (data_points,))
        w = ols(X, y, 'qr')
        self.assertArraysAllClose(w, true_w, atol=1e-2)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
