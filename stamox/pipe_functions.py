import jax.numpy as jnp

from .anova import one_way
from .basic import scale
from .cluster import kmeans
from .core import (
    make_partial_pipe,
    make_pipe,
    predict,
    summary,
)
from .correlation import cor, pearsonr, spearmanr
from .decomposition import princomp
from .distribution import (
    dbeta,
    dbinom,
    dcauchy,
    dchisq,
    dexp,
    dF,
    dgamma,
    dlaplace,
    dnorm,
    dpareto,
    dpoisson,
    dt,
    dunif,
    dweibull,
    ecdf,
    pbeta,
    pbinom,
    pcauchy,
    pchisq,
    pexp,
    pF,
    pgamma,
    plaplace,
    pnorm,
    ppareto,
    ppoisson,
    pt,
    punif,
    pweibull,
    qbeta,
    qbinom,
    qcauchy,
    qchisq,
    qexp,
    qF,
    qgamma,
    qlaplace,
    qnorm,
    qpareto,
    qpoisson,
    qt,
    qunif,
    qweibull,
    rbeta,
    rbinom,
    rcauchy,
    rchisq,
    rexp,
    rF,
    rgamma,
    rlaplace,
    rnorm,
    rpareto,
    rpoisson,
    rt,
    runif,
    rweibull,
)
from .hypothesis import (
    bartlett_test,
    durbin_watson_test,
    friedman_test,
)
from .regression import lm
from .sample import bootstrap, bootstrap_sample, jackknife, jackknife_sample
from .transformation import boxcox, z_fisher


one_way = make_partial_pipe(one_way, name="one_way")
kmeans = make_partial_pipe(kmeans, name="kmeans")
spearmanr = make_partial_pipe(spearmanr, name="spearmanr")
pearsonr = make_partial_pipe(pearsonr, name="pearsonr")
cor = make_partial_pipe(cor, name="cor")
bartlett_test = make_partial_pipe(bartlett_test, name="bartlett_test")
durbin_watson_test = make_partial_pipe(durbin_watson_test, name="durbin_watson_test")
friedman_test = make_partial_pipe(friedman_test, name="friedman_test")
bootstrap = make_partial_pipe(bootstrap, name="bootstrap")
bootstrap_sample = make_partial_pipe(bootstrap_sample, name="bootstrap_sample")
jackknife = make_partial_pipe(jackknife, name="jackknife")
jackknife_sample = make_pipe(jackknife_sample, name="jackknife_sample")
boxcox = make_partial_pipe(boxcox, name="boxcox")
z_fisher = make_partial_pipe(z_fisher, name="z_fisher")
lm = make_partial_pipe(lm, name="lm")
dbeta = make_partial_pipe(dbeta, name="dbeta")
dbinom = make_partial_pipe(dbinom, name="dbinom")
dcauchy = make_partial_pipe(dcauchy, name="dcauchy")
dchisq = make_partial_pipe(dchisq, name="dchisq")
dexp = make_partial_pipe(dexp, name="dexp")
dF = make_partial_pipe(dF, name="dF")
dgamma = make_partial_pipe(dgamma, name="dgamma")
dlaplace = make_partial_pipe(dlaplace, name="dlaplace")
dnorm = make_partial_pipe(dnorm, name="dnorm")
dpareto = make_partial_pipe(dpareto, name="dpareto")
dpoisson = make_partial_pipe(dpoisson, name="dpoisson")
dt = make_partial_pipe(dt, name="dt")
dunif = make_partial_pipe(dunif, name="dunif")
dweibull = make_partial_pipe(dweibull, name="dweibull")
ecdf = make_partial_pipe(ecdf, name="ecdf")
pbeta = make_partial_pipe(pbeta, name="pbeta")
pbinom = make_partial_pipe(pbinom, name="pbinom")
pcauchy = make_partial_pipe(pcauchy, name="pcauchy")
pchisq = make_partial_pipe(pchisq, name="pchisq")
pexp = make_partial_pipe(pexp, name="pexp")
pF = make_partial_pipe(pF, name="pF")
pgamma = make_partial_pipe(pgamma, name="pgamma")
plaplace = make_partial_pipe(plaplace, name="plaplace")
pnorm = make_partial_pipe(pnorm, name="pnorm")
ppareto = make_partial_pipe(ppareto, name="ppareto")
ppoisson = make_partial_pipe(ppoisson, name="ppoisson")
pt = make_partial_pipe(pt, name="pt")
punif = make_partial_pipe(punif, name="punif")
pweibull = make_partial_pipe(pweibull, name="pweibull")
qbeta = make_partial_pipe(qbeta, name="qbeta")
qbinom = make_partial_pipe(qbinom, name="qbinom")
qcauchy = make_partial_pipe(qcauchy, name="qcauchy")
qchisq = make_partial_pipe(qchisq, name="qchisq")
qexp = make_partial_pipe(qexp, name="qexp")
qF = make_partial_pipe(qF, name="qF")
qgamma = make_partial_pipe(qgamma, name="qgamma")
qlaplace = make_partial_pipe(qlaplace, name="qlaplace")
qnorm = make_partial_pipe(qnorm, name="qnorm")
qpareto = make_partial_pipe(qpareto, name="qpareto")
qpoisson = make_partial_pipe(qpoisson, name="qpoisson")
qt = make_partial_pipe(qt, name="qt")
qunif = make_partial_pipe(qunif, name="qunif")
qweibull = make_partial_pipe(qweibull, name="qweibull")
rbeta = make_partial_pipe(rbeta, name="rbeta")
rbinom = make_partial_pipe(rbinom, name="rbinom")
rcauchy = make_partial_pipe(rcauchy, name="rcauchy")
rchisq = make_partial_pipe(rchisq, name="rchisq")
rexp = make_partial_pipe(rexp, name="rexp")
rF = make_partial_pipe(rF, name="rF")
rgamma = make_partial_pipe(rgamma, name="rgamma")
rlaplace = make_partial_pipe(rlaplace, name="rlaplace")
rnorm = make_partial_pipe(rnorm, name="rnorm")
rpareto = make_partial_pipe(rpareto, name="rpareto")
rpoisson = make_partial_pipe(rpoisson, name="rpoisson")
rt = make_partial_pipe(rt, name="rt")
runif = make_partial_pipe(runif, name="runif")
rweibull = make_partial_pipe(rweibull, name="rweibull")
predict = make_partial_pipe(predict, name="predict")
summary = make_partial_pipe(summary, name="summary")
mean = make_partial_pipe(jnp.mean, "mean")
sd = make_partial_pipe(jnp.std, "sd")
var = make_partial_pipe(jnp.var, "var")
median = make_partial_pipe(jnp.median, "median")
quantile = make_partial_pipe(jnp.quantile, "quantile")
min = make_partial_pipe(jnp.min, "min")
max = make_partial_pipe(jnp.max, "max")
sum = make_partial_pipe(jnp.sum, "sum")
prod = make_partial_pipe(jnp.prod, "prod")
cumsum = make_partial_pipe(jnp.cumsum, "cumsum")
cumprod = make_partial_pipe(jnp.cumprod, "cumprod")
diff = make_partial_pipe(jnp.diff, "diff")
scale = make_partial_pipe(scale, "scale")
princomp = make_partial_pipe(princomp, "princomp")

__all__ = [
    "one_way",
    "kmeans",
    "spearmanr",
    "pearsonr",
    "cor",
    "bartlett_test",
    "durbin_watson_test",
    "friedman_test",
    "lm",
    "bootstrap",
    "bootstrap_sample",
    "jackknife",
    "jackknife_sample",
    "boxcox",
    "z_fisher",
    "dbeta",
    "dbinom",
    "dcauchy",
    "dchisq",
    "dexp",
    "dF",
    "dgamma",
    "dlaplace",
    "dnorm",
    "dpareto",
    "dpoisson",
    "dt",
    "dunif",
    "dweibull",
    "ecdf",
    "pbeta",
    "pbinom",
    "pcauchy",
    "pchisq",
    "pexp",
    "pF",
    "pgamma",
    "plaplace",
    "pnorm",
    "ppareto",
    "ppoisson",
    "pt",
    "punif",
    "pweibull",
    "qbeta",
    "qbinom",
    "qcauchy",
    "qchisq",
    "qexp",
    "qF",
    "qgamma",
    "qlaplace",
    "qnorm",
    "qpareto",
    "qpoisson",
    "qt",
    "qunif",
    "qweibull",
    "rbeta",
    "rbinom",
    "rcauchy",
    "rchisq",
    "rexp",
    "rF",
    "rgamma",
    "rlaplace",
    "rnorm",
    "rpareto",
    "rpoisson",
    "rt",
    "runif",
    "rweibull",
    "predict",
    "summary",
    "mean",
    "sd",
    "var",
    "median",
    "quantile",
    "min",
    "max",
    "sum",
    "prod",
    "cumsum",
    "cumprod",
    "diff",
    "scale",
    "princomp",
]
