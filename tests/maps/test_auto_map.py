"""Test for Auto Map"""
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest


from stamox.maps import auto_map


class AutoMapTest(jtest.JaxTestCase):

    def test_single_scalar_input(self):
        a = 2.0
        
        def f(x):
            return x + 1
        
        b = auto_map(f, a)
        self.assertAllClose(b, np.array(3.0))

    def test_single_vector_input(self):
        a = np.array([1, 2, 3])
        def f(x):
            return x + 1
        b = auto_map(f, a)
        self.assertAllClose(b, a + 1)

    def test_multiple_scalars_inputs(self):
        a = 1.
        b = 2.
        c = 3.

        def f(x, y, z):
            return x*y*z

        d = auto_map(f, a, b, c)
        self.assertAllClose(d, jnp.array(6.))

    def test_scalar_and_vector_inputs(self):
        a = 1.
        b = jnp.array([1. ,2. , 3.])
        def f(x, y):
            return x * y

        c = auto_map(f, b, a)
        self.assertAllClose(c, b)
    

    def test_equal_size_vectors_inputs(self):
        a = jnp.array([1. ,2. , 3.])
        b = jnp.array([1. ,2. , 3.])

        def f(x, y):
            return x * y

        c = auto_map(f, b, a)
        self.assertAllClose(c, jnp.array([1. ,4. , 9.]))
    
    def test_unequal_size_vectors_inputs(self):
        a = jnp.array([1. ,2. ])
        b = jnp.array([1. ,2. , 3.])
        def f(x, y):
            return x * y

        c = auto_map(f, b, a)
        self.assertAllClose(c, jnp.array([1. ,4. , 3.]))
    
    def test_scalar_and_unequal_size_vectors_inputs(self):

        a = jnp.array([1. ,2. ])
        b = jnp.array([1. ,2. , 3.])
        c = 1.

        def f(x, y, z):
            return x * y + z

        c = auto_map(f, b, a, c)
        self.assertAllClose(c, jnp.array([2. ,5. , 4.]))

    def test_scalar_and_unequal_size_vectors_inputs_and_jit_compatible(self):

        a = jnp.array([1., 2.])
        b = jnp.array([1., 2., 3.])
        c = 1.

        @jit
        def f(x, y, z):
            return x + y * z

        c = auto_map(f, c, a, b)
        self.assertAllClose(c, jnp.array([1., 4., 3.]) + 1)

    def test_scalar_value_and_grad_compatible(self):

        a = 2.0

        @value_and_grad
        def f(x):
            return x + 1

        b, b_ = auto_map(f, a)

        self.assertAllClose(b, 3.)
        self.assertAllClose(b_, 1.)

    def test_scalar_grad_compatible(self):

        a = 2.0

        @grad
        def f(x):
            return x + 1

        b = auto_map(f, a)
        self.assertAllClose(b, np.array(1.0))


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())