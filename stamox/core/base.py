from typing import Callable, Optional
from functools import partial

import equinox as eqx
from jaxtyping import PyTree

from .pipe import Pipe


class Functional(eqx.Module):
    """General Function"""

    _fn: Callable

    def __init__(self, fn: Optional[Callable | None] = None):
        """Make a General Function

        Args:
            fn (Optional[Callable|None], optional): Callable object.
        """
        super().__init__()
        self._fn = fn

    def desc(self):
        """Description for the function"""
        pass

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            raise ValueError("No Callable Function to Call")
        return self._fn(*args, **kwargs)

    def __rshift__(self, _next) -> Pipe:
        """Make Pipe"""
        return Pipe([self, _next])


class StateFunc(Functional):
    """Functions with State"""

    _name: str
    _params: PyTree

    def __init__(
        self,
        params: Optional[PyTree | None] = None,
        name: Optional[str | None] = "Anonymous",
        fn: Optional[Callable | None] = None,
    ):
        """Initialize a Stateful Function

        Args:
            params (Optional[PyTree|None], optional): Function Params. Defaults to None.
            name (Optional[str|None], optional): Function Name. Defaults to "Anonymous".
            fn (Optional[Callable|None], optional): Given Function. Defaults to None.
        """
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
    """Functions without State"""

    _name: str

    def __init__(
        self,
        name: Optional[str | None] = "Anonymous",
        fn: Optional[Callable | None] = None,
    ):
        """Initialize a Stateless Function

        Args:
            name (Optional[str|None], optional): Function Name. Defaults to "Anonymous".
            fn (Optional[Callable|None], optional): Given Function. Defaults to None.
        """
        super().__init__(fn=fn)
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def make_pipe(cls: Callable, params: PyTree = None, name: str = "Anonymous"):
    """Make a Function Pipable

    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls):
        if params is None:
            return StatelessFunc(name=name, fn=cls)
        else:
            return StateFunc(params=params, name=name, fn=cls)

    if cls is None:
        return wrap

    return wrap(cls)


def make_partial_pipe(cls: Callable, params: PyTree = None, name: str = "Anonymous"):
    """Make a Partial Function Pipe
    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """
    def wrap(cls):
        def partial_fn(**kwargs):
            fn = partial(cls, **kwargs)
            if params is None:
                return StatelessFunc(name=name, fn=fn)
            else:
                return StateFunc(params=params, name=name, fn=fn)

        return partial_fn

    if cls is None:
        return wrap

    return wrap(cls)
