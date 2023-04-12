from .base import StateFunc
from .pipe import make_partial_pipe, make_pipe


@make_partial_pipe
def predict(x, state: StateFunc):
    return state._predict(x)


@make_pipe
def summary(state: StateFunc):
    return state._summary()
