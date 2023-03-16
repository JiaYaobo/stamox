from typing import Any
import jax
import equinox


import pandas as pd
from pandas import DataFrame


class DataX(DataFrame):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    

    
    

