from functools import wraps
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
        >>> from stamox import pipe_jit
        >>> f = lambda x: x + 1
        >>> f = pipe_jit(f)
        >>> g = f >> f >> f
        >>> g(1)
        4
    """
    if name is None and func is not None:
        if hasattr(func, "name"):
            name = func.name
        elif hasattr(func, "__name__"):
            name = func.__name__
        else:
            name = "none"

    @wraps(func)
    def wrap(func: Callable[..., T]) -> Callable:
        fn = filter_jit(func, donate=donate)

        return Functional(name=name, fn=fn, pipe_type="jit")

    return wrap if func is None else wrap(func)
