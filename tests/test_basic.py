from stamox.core import make_partial_pipe
from equinox import filter_grad, filter_jit
import jax.numpy as jnp

@make_partial_pipe
@filter_jit
@filter_grad
def f(x, y):
    return y * x ** 3
       
print(f(y=3.)(jnp.array(1.)))