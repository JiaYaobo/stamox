from .base import Functional, StateFunc


class summary(Functional):
    def __call__(self, x: StateFunc, *args, **kwargs):
        return x._summary()