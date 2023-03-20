from typing import Any, Optional

import equinox as eqx
import jax
import jax.random as jrandom


class Pipe(eqx.nn.Sequential):
    def __init__(self, funcs) -> None:
        super().__init__(funcs)

    def __call__(self, x: Any, *, key: Optional["jax.random.PRNGKey"] = None):
        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jrandom.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            x = layer(x, key=key)
        return x

    def __rshift__(self, _next):
        return Pipe([*self.layers, _next])
