import jax
import jax.numpy as jnp
import jax.random as jr
import random
import equinox as eqx
import pandas as pd
import polars as pl



def _boot_splits(data: pl.DataFrame, times=25, strata=None, breaks=4, pool=0.1):
    n = data.shape[0]

    if strata == None:
        pass


class Bootstraps(eqx.Module):

    def __init__(self, data, times=25, strata=None, breaks=4, pool=0.1, *args, **kwargs) -> None:
        super().__init__()
