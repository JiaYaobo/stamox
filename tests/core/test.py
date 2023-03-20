import stamox as stx
import equinox as eqx
import jax.numpy as jnp

@stx.core.make_pipable
def f(x, key=None):
    return x + 1


wei = {'w': jnp.array([5.])}

def k(x, key=None):
    return x * wei['w']

k = stx.core.make_pipable(k, params=wei)


g = stx.core.StatelessFunc(fn=lambda x, key=None: x * 2)



h2 = f >> g >> g >> f >> g >> k >> f


print(h2(1))
print(k(2))