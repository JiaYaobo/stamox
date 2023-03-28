from functools import partial
from typing import Any, Callable

from equinox import filter_jit

from .base import Functional


def pipe_jit(cls: Callable, name: str = "Anonymous", **kwargs) -> Callable:
    """Make a Function Pipe Jitted

    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls):
        fn = filter_jit(cls)
        return Functional(name=name, fn=fn)

    if cls is None:
        return wrap

    return wrap(cls)


def partial_pipe_jit(cls: Callable, name: str = "Anonymous", **kwargs) -> Callable:
    """Make a Partial Function Pipe Jitted
    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls) -> Callable:
        def partial_fn(x: Any = None, *args, **kwargs):
            fn = filter_jit(cls)
            fn = partial(fn, **kwargs)
            if x is not None:
                return fn(x, *args, **kwargs)
            return Functional(name=name, fn=fn)

        return Functional(name="partial_jitted_" + name, fn=partial_fn)

    if cls is None:
        return wrap

    return wrap(cls)
