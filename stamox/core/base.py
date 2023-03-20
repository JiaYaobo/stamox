from typing import Callable, Optional

import equinox as eqx
from jaxtyping import PyTree

from .pipe import Pipe


class Functional(eqx.Module):
    _fn: Callable

    def __init__(self, fn: Optional[Callable | None] = None):
        super().__init__()
        self._fn = fn
    
    def desc(self):
        pass

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            raise ValueError("No Callable Function to Call")
        return self._fn(*args, **kwargs)

    def __rshift__(self, _next) -> Pipe:
        return Pipe([self, _next])


class StateFunc(Functional):
    _name: str
    _params: PyTree

    def __init__(
        self,
        params: Optional[PyTree | None] = None,
        name: Optional[str | None] = "Anonymous",
        fn: Optional[Callable | None] = None,
    ):
        super().__init__(fn=fn)
        self._name = name
        self._params = params

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> PyTree:
        return self._params

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class StatelessFunc(Functional):
    _name: str

    def __init__(
        self,
        name: Optional[str | None] = "Anonymous",
        fn: Optional[Callable | None] = None,
    ):
        super().__init__(fn=fn)
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def make_pipable(cls: Callable, params: PyTree = None, name: str = "Anonymous"):
    def wrap(cls):
        if params is None:
            return StatelessFunc(name=name, fn=cls)
        else:
            return StateFunc(params=params, name=name, fn=cls)

    if cls is None:
        return wrap

    return wrap(cls)
