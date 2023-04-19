from .core import make_partial_pipe, make_pipe
from .correlation import cor, pearsonr, spearmanr
from .sample import bootstrap, bootstrap_sample, jackknife, jackknife_sample
from .transformation import boxcox, z_fisher


spearmanr = make_partial_pipe(spearmanr, name="spearmanr")
pearsonr = make_partial_pipe(pearsonr, name="pearsonr")
cor = make_partial_pipe(cor, name="cor")
bootstrap = make_partial_pipe(bootstrap, name="bootstrap")
bootstrap_sample = make_partial_pipe(bootstrap_sample, name="bootstrap_sample")
jackknife = make_partial_pipe(jackknife, name="jackknife")
jackknife_sample = make_pipe(jackknife_sample, name="jackknife_sample")
boxcox = make_partial_pipe(boxcox, name="boxcox")
z_fisher = make_pipe(z_fisher, name="z_fisher")


__all__ = [
    "spearmanr",
    "pearsonr",
    "cor",
    "bootstrap",
    "bootstrap_sample",
    "jackknife",
    "jackknife_sample",
    "boxcox",
    "z_fisher",
]
