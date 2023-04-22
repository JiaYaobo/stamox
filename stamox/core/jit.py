from functools import partial, wraps
from typing import Callable, TypeVar

from equinox import filter_jit

from .base import Functional


T = TypeVar("T")


def pipe_jit(
    func: Callable[..., T] = None, *, donate: str = "none", name: str = None
) -> Callable[..., T]:
    """Creates a pipeable jitted functional from a given function.

    Args:
        func: The function to create the functional from.
        donate: Optional donation string.
        name: Optional name for the functional.

    Returns:
        A callable that creates a functional from the given function.

    Example:
        >>> from stamox.core import pipe_jit
        >>> f = lambda x: x + 1
        >>> f = pipe_jit(f)
        >>> g = f >> f >> f
        >>> g(1)
        4
    """
    if name is None and func is not None:
        if hasattr(func, "name"):
            name = func.name
        else:
            name = func.__name__

    @wraps(func)
    def wrap(func: Callable[..., T]) -> Callable:
        fn = filter_jit(func, donate=donate)

        return Functional(name=name, fn=fn)

    return wrap if func is None else wrap(func)


def partial_pipe_jit(
    func: Callable[..., T] = None, *, name: str = None
) -> Callable[..., T]:
    """Creates a partial pipeable jitted functional from a given function.

    Args:
        func (Callable[P, T]): _description_
        name (str, optional): _description_. Defaults to None.

    Returns:
        a partial pipeable jitted functional from a given function.

    Example:
        >>> from stamox.core import partial_pipe_jit
        >>> f = lambda x, y: x + y
        >>> f = partial_pipe_jit(f)
        >>> g = f(y=1) >> f(y=2) >> f(y=3)
        >>> g(1)
        7
    """
    if name is None and func is not None:
        if hasattr(func, "name"):
            name = func.name
        else:
            name = func.__name__

    @wraps(func)
    def wrap(func: Callable[..., T]) -> Callable:
        @wraps(func)
        def partial_fn(*args, donate: str = "none", **kwargs):
            fn = filter_jit(func, donate=donate)
            fn = partial(fn, **kwargs)
            if len(args) != 0:
                return fn(*args, **kwargs)
            return Functional(name=name, fn=fn)

        return partial_fn

    return wrap if func is None else wrap(func)
