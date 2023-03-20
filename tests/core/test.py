import stamox as stx

from jax import grad, jit
import jax.random as jrandom

key = jrandom.PRNGKey(1)


@stx.core.make_partial_pipe
@grad
def f(x, key):
    return 2 * x + jrandom.normal(key, shape=())

@stx.core.make_pipe
@jit
@grad
def g(x):
    return x + 1.

@stx.core.make_pipe
@grad
def h(x):
    return x ** 2


pipe = g >> f(key=key) >> h

print(pipe(1.))


# f = ft.partial(f, key=jrandom.PRNGKey(0))
# g = ft.partial(g, key=jrandom.PRNGKey(0))
# h = ft.partial(h, key=jrandom.PRNGKey(0))

# f = stx.core.make_pipe(f, name='f')
# g = stx.core.make_pipe(g, name='g')
# h = stx.core.make_pipe(h, name='h')



# pipe = f >> g >> h

