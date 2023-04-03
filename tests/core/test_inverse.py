"""Test Inverse Function"""
import jax.numpy as jnp
from absl.testing import absltest
from equinox import filter_make_jaxpr
from jax._src import test_util as jtest
from jax import make_jaxpr

# from stamox.core import Functional, inverse


class InverseTest(jtest.JaxTestCase):
    def test_inverse(self, *args, **kwargs):
        def f(x):
            return jnp.exp(x)

        print(filter_make_jaxpr(f)(1.)[0])
        print(make_jaxpr(f)(1.).jaxpr)
        print(filter_make_jaxpr(f)(1.)[2])
        print(make_jaxpr(f)(1.).literals)

if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())