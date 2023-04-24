from .base import StateFunc


def predict(x, state: StateFunc):
    return state._predict(x)


def summary(state: StateFunc):
    return state._summary()
