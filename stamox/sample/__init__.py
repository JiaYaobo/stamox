from stamox.core import make_pipe
from stamox.sample._bootstrap import bootstrap, bootstrap_sample
from stamox.sample._jackknife import jackknife, jackknife_sample_fun


jackknife_sample = make_pipe(jackknife_sample_fun, name="jackknife_sample")

__all__ = [
    "bootstrap",
    "bootstrap_sample",
    "jackknife",
    "jackknife_sample",
]
