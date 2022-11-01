import jax.random as jrand
import jax.numpy as jnp
import numpy as np

from sax.distrix import pt, qt, rt


def test_rt():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000, 10)
    df = 2
    ts = rt(key, df, sample_shape=sample_shape)
    return (ts.mean(axis=0), ts.var(axis=0))


def test_pt():
    x = jnp.array([-1.96, -1.645, -1., 0, 1., 1.645, 1.96])
    df = 2
    p = pt(x, df)
    return p

def test_qt():
    q = [0.02499789, 0.04998492, 0.15865527,0.5 ,0.8413447, 0.95001507, 0.9750021]
    df = 2
    x = qt(q, df)
    return x

print(test_rt())
print(test_pt())
print(test_qt())