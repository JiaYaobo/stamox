"""Test for student t distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np

from stamox.distrix import pt, rt, qt, dt

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest


class TTest(jtest.JaxTestCase):

    def test_rt(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000, )
        df = 6
        ts = rt(key, df, 0., 1., sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, 0., atol=1e-2)
        self.assertAllClose(var, float(df / (df - 2)), atol=1e-2)

    def test_pt(self):
        x = np.array([1., 1.645, 1.96, 2.65, 3.74])
        p = pt(x, 6)
        true_p = np.array(
            [0.8220412, 0.9244638, 0.9511524, 0.9809857, 0.9951888])
        self.assertArraysAllClose(p, true_p)

    def test_qt(self):
        q = [0.8220412, 0.9244638, 0.9511524, 0.9809857, 0.9951888]
        x = qt(q, 6)
        true_x = np.array([1., 1.645, 1.96, 2.65, 3.74])
        self.assertArraysAllClose(x, true_x)

    def test_dt(self):
        x = np.array([1., 1.645, 1.96, 2.65, 3.74])
        grads = dt(x, 6)
        true_grads = np.array(
            [0.223142291, 0.104005269, 0.067716584, 0.025409420, 0.005672347])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
