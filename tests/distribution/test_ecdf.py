"""Test the ECDF."""
import jax.numpy as jnp
import numpy as np

import stamox.pipe_functions as PF
from stamox.distribution import ecdf


def test_ecdf():
    x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = ecdf(x)(x)
    true_y = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(y, true_y)

def test_pipe_ecdf():
    x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = PF.ecdf()(x)(x)
    true_y = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(y, true_y)
