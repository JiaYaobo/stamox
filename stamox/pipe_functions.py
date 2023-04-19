from .cluster import kmeans
from .core import make_partial_pipe, make_pipe
from .correlation import cor, pearsonr, spearmanr
from .hypothesis import (
    bartlett_test,
    durbin_watson_test,
    friedman_test,
    shapiro_wilk_test,
)
from .regression import lm
from .sample import bootstrap, bootstrap_sample, jackknife, jackknife_sample
from .transformation import boxcox, z_fisher


kmeans = make_partial_pipe(kmeans, name="kmeans")
spearmanr = make_partial_pipe(spearmanr, name="spearmanr")
pearsonr = make_partial_pipe(pearsonr, name="pearsonr")
cor = make_partial_pipe(cor, name="cor")
bartlett_test = make_pipe(bartlett_test, name="bartlett_test")
durbin_watson_test = make_partial_pipe(durbin_watson_test, name="durbin_watson_test")
friedman_test = make_pipe(friedman_test, name="friedman_test")
shapiro_wilk_test = make_pipe(shapiro_wilk_test, name="shapiro_wilk_test")
bootstrap = make_partial_pipe(bootstrap, name="bootstrap")
bootstrap_sample = make_partial_pipe(bootstrap_sample, name="bootstrap_sample")
jackknife = make_partial_pipe(jackknife, name="jackknife")
jackknife_sample = make_pipe(jackknife_sample, name="jackknife_sample")
boxcox = make_partial_pipe(boxcox, name="boxcox")
z_fisher = make_pipe(z_fisher, name="z_fisher")
lm = make_partial_pipe(lm, name="lm")


__all__ = [
    "kmeans",
    "spearmanr",
    "pearsonr",
    "cor",
    "bartlett_test",
    "durbin_watson_test",
    "friedman_test",
    "lm",
    "shapiro_wilk_test",
    "bootstrap",
    "bootstrap_sample",
    "jackknife",
    "jackknife_sample",
    "boxcox",
    "z_fisher",
]
