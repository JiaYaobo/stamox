from .base import Functional, StateFunc


class Summary(Functional):
    def __call__(self, x: StateFunc, *args, **kwargs):
        return print(x._summary())
    
summary = Summary()