import jax
import equinox


import pandas as pd


class Datax(pd.DataFrame):

    def __init__(self, *args, **kwargs) -> None:
        super(Datax, self).__init__(*args, **kwargs)
    

    
    

