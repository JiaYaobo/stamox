import jax
import equinox


import pandas as pd
from pandas import DataFrame


class DataX(DataFrame):

    def __init__(self, *args, **kwargs) -> None:
        super(DataX, self).__init__(*args, **kwargs)    
        self._fun_tree = []
    def __rshift__(self, func):
        self._fun_tree.append(func)
    

    
    

