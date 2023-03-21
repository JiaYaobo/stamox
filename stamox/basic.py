import jax.numpy as jnp
from .core import StateFunc, StatelessFunc


class mean(StatelessFunc):
    def __init__(self):
        super().__init__(name='mean', fn=jnp.mean)
    
    def __call__(cls, x: Any, axis=0):
        pass
