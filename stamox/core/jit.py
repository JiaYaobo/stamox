from functools import partial, wraps
from typing import Any, Callable, ParamSpec, TypeVar

from equinox import filter_jit

from .base import Functional


P = ParamSpec("P")
T = TypeVar("T")


def pipe_jit(
    func: Callable[P, T], *, donate: str = "none", name: str = "Anonymous"
) -> Callable[P, T]:
    """Make a Function Pipe Jitted

    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    if name is None and func is not None:
        name = func.__name__

    @wraps(func)
    def wrap(func: Callable[P, T]) -> Callable:
        fn = filter_jit(func, donate=donate)

        @wraps(func)
        def create_functional(*args, **kwargs):
            return Functional(name=name, fn=fn)(*args, **kwargs)
        return create_functional

    return wrap if func is None else wrap(func)


def partial_pipe_jit(
    func: Callable[P, T], *, name: str = "Anonymous"
) -> Callable[P, T]:
    """Make a Partial Function Pipe Jitted
    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """
    if name is None and func is not None:
        name = func.__name__

    @wraps(func)
    def wrap(func: Callable[P, T]) -> Callable:

        @wraps(func)
        def partial_fn(x: Any = None, *args, donate: str = "none", **kwargs):
            fn = filter_jit(func, donate=donate)
            fn = partial(fn, **kwargs)
            if x is not None:
                return fn(x, *args, **kwargs)
            return Functional(name=name, fn=fn)

        return partial_fn

    return wrap if func is None else wrap(func)
