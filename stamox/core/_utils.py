import inspect
from typing import Callable, Dict


def _check_if_fully_partial(func: Callable, kwargs: Dict) -> bool:
    """Check if a function is fully partial.

    Args:
        func (Callable): Function to check.
        kwargs (Dict): Keyword arguments for the function.

    Returns:
        bool: True if the function is fully partial, False otherwise.
    """
    first_argname = inspect.getfullargspec(func).args[0]
    if first_argname in kwargs.keys():
        return True
    return False