from ._gamma import pgamma, qgamma, rgamma, dgamma


def dchisq(x, df=2.):
    return dgamma(x, shape=df/2, rate=1/2)


def pchisq(x, df=2.):
    return pgamma(x, shape=df/2, rate=1/2)


def qchisq(q, df):
    return qgamma(q, shape=df/2, rate=1/2)


def rchisq(key, df=2., sample_shape=()):
    return rgamma(key, shape=df / 2, rate=1 / 2, sample_shape=sample_shape)
