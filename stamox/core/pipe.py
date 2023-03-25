from typing import Tuple, Sequence, Union, Any, Callable, Optional
from functools import partial

import equinox as eqx

from .base import Functional


class Pipe(eqx.Module):
    """A class for creating a pipe of functions.

    Attributes:
        funcs (Tuple[Functional, ...]): A tuple of Functional objects.
    """

    funcs: Tuple[Functional, ...]

    def __init__(self, funcs: Sequence[Functional]) -> None:
        """Initialize the Pipe object.

        Args:
            funcs (Sequence[Functional]): A sequence of Functional objects.
        """
        self.funcs = tuple(funcs)

    def __call__(self, x: Any = None, *args, **kwargs):
        """Call the Pipe object.

        Args:
            x (Any): The input to the Pipe object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The output of the Pipe object.
        """
        for fn in self.funcs:
            if isinstance(fn, Pipeable):
                x = fn()
            x = fn(x, *args, **kwargs)
        return x

    def __getitem__(self, i: Union[int, slice, str]) -> Functional:
        """Get an item from the Pipe object.

        Args:
            i (Union[int, slice, str]): The index or name of the item.

        Returns:
            Functional: The item at the given index or with the given name.

        Raises:
            TypeError: If the type of the index is not supported.
            ValueError: If no function names the given string.
        """
        if isinstance(i, int):
            return self.funcs[i]
        elif isinstance(i, slice):
            return Pipe(self.funcs[i])
        elif isinstance(i, str):
            _f = []
            i = i.lower()
            for f in self.funcs:
                if f.name.lower() == i:
                    _f.append(f)

            if len(_f) == 0:
                raise ValueError(f"No Function Names {i}")
            if len(_f) > 1:
                return Pipe(_f)
            else:
                return _f[0]
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self) -> Functional:
        """Iterate over the Pipe object.

        Yields:
            Functional: The next item in the Pipe object.
        """
        yield from self.funcs

    def __len__(self) -> int:
        """Get the length of the Pipe object.

        Returns:
            int: The length of the Pipe object.
        """
        return len(self.funcs)

    def __rshift__(self, _next):
        """Create a new Pipe object by appending another item.

        Args:
            _next (Any): The item to append.

        Returns:
            Pipe: A new Pipe object with the given item appended.
        """
        if not isinstance(_next, Functional):
            _next = Functional(fn=_next)
        if isinstance(_next, Pipe):
            return Pipe([*self.funcs, *_next])
        return Pipe([*self.funcs, _next])


class Pipeable(Functional):
    """A class for pipeable functions.

    Attributes:
        value (Any): The value to be piped.
    """

    value: Any

    def __init__(self, value):
        super().__init__(name="PipeableData", fn=None)
        """Initialize the Pipeable object.

        Args:
            value (Any): The value to be piped.
        """
        self.value = value

    def __call__(self, *args, **kwargs):
        """Pipe the value through the function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The piped value.
        """
        return self.value


def make_pipe(
    cls: Optional[Callable] = None, name: str = "PipeableFunc", **kwargs
) -> Callable:
    """Makes a Function Pipeable.

    Args:
        cls (Callable): Function or Callable Class.
        name (str, optional): Name of the Function. Defaults to "PipeableFunc".
        kwargs (optional): Additional keyword arguments.

    Returns:
        Callable: The wrapped function.
    """

    def wrap(cls):
        return Functional(name=name, fn=cls)

    if cls is None:
        return wrap

    return wrap(cls)


def make_partial_pipe(
    cls: Optional[Callable] = None, name: str = "PipeableFunc", **kwargs
) -> Callable:
    """Makes a Partial Function Pipe.

    Args:
        cls (Callable): Function or Callable Class.
        name (str, optional): Name of the Function. Defaults to "PipeableFunc".
        kwargs (dict): Keyword arguments for the function.

    Returns:
        Callable: A partial function pipe.
    """

    def wrap(cls) -> Callable:
        def partial_fn(x: Any = None, *args, **kwargs):
            fn = partial(cls, **kwargs)
            if x is not None:
                return fn(x, *args, **kwargs)
            return Functional(name=name, fn=fn)

        return Functional(name="partial_" + name, fn=partial_fn)

    if cls is None:
        return wrap

    return wrap(cls)
