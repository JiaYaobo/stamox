from typing import Callable, Optional, Any
from functools import partial

import equinox as eqx
from jaxtyping import PyTree


class Functional(eqx.Module):
    """General Function"""

    _fn: Callable

    def __init__(self, fn: Optional[Callable] = None):
        """Make a General Function

        Args:
            fn (Optional[Callable|None], optional): Callable object.
        """
        super().__init__()
        self._fn = fn

    def desc(self):
        """Description for the function"""
        pass

    def __call__(self, x: Any, *args, **kwargs):
        if self._fn is None:
            raise ValueError("No Callable Function to Call")
        return self._fn(x, *args, **kwargs)

    def __rshift__(self, _next):
        """Make Pipe"""
        from .pipe import Pipe

        return Pipe([self, _next])


class StateFunc(Functional):
    """Functions with State"""

    _name: str
    _params: PyTree

    def __init__(
        self,
        *,
        name: Optional[str] = "Anonymous",
        fn: Optional[Callable] = None,
        **params
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

    def __call__(self, x: Any, *args, **kwargs):
        return super().__call__(x, *args, **kwargs)


class StatelessFunc(Functional):
    """Functions without State"""

    _name: str

    def __init__(
        self, *, name: Optional[str] = "Anonymous", fn: Optional[Callable] = None
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

    def __call__(self, x: Any, *args, **kwargs):
        return super().__call__(x, *args, **kwargs)


def make_pipe(
    cls: Callable,
    params: PyTree = None,
    name: str = "Anonymous",
) -> Callable:
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


def make_partial_pipe(
    cls: Callable, params: PyTree = None, name: str = "Anonymous"
) -> Callable:
    """Make a Partial Function Pipe
    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls) -> Callable:
        def partial_fn(x: Any = None, **kwargs):
            fn = partial(cls, **kwargs)
            if x is not None:
                return fn(x)
            if params is None:
                return StatelessFunc(name=name, fn=fn)
            else:
                return StateFunc(params=params, name=name, fn=fn)

        return partial_fn

    if cls is None:
        return wrap

    return wrap(cls)
