from .base import Functional, StateFunc


class Predict(Functional):
    def __call__(self, x: StateFunc, *args, **kwargs):
        return x._predict(*args, **kwargs)


class Summary(Functional):
    def __call__(self, x: StateFunc, *args, **kwargs):
        return print(x._summary(*args, **kwargs))


summary = Summary()
predict = Predict()