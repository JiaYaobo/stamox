# pruns <- function(q, n1, n2, lower.tail = TRUE, log.p = FALSE){
#   stopifnot(is.numeric(q) & n1>0 & n2>0)
#   q <- ifelse(q >= 1, q, 1)
#   q <- ifelse(q <= n1+n2, q, n1+n2)
#   q <- round(q)
#   tmp <- cumsum(druns(1:max(q),n1,n2,log=log.p))
#   r0 <- tmp[q]
#   if (lower.tail==FALSE){r0<- 1-r0}  
# #  r0 <- NULL
# #  if (lower.tail){
# #    for (i in 1:length(q)){r0 <- c(r0,ifelse(q[i]>=2,sum(druns(x=2:floor(q[i]),n1,n2,log=log)),0))}
# #  }
# #  else {r0 <- 1-pruns(q,n1,n2,lower.tail=T, log=log)}  
#   return(r0)  
# }  
# ##
# ##  Quantile function of the runs statistic
# ##
# qruns <- function(p, n1, n2, lower.tail = TRUE, log.p = FALSE){
#   r0 <- NULL
#   q1 <- ifelse (n1==n2, 2*n1, 2*min(n1,n2)+1) 
#   pr <- c(0, cumsum(druns(2:q1, n1, n2)))
#   for (i in 1:length(p)){
#     if (p[i]>=0 & p[i]<=1){
#       #rq<-which(abs(pr-p)==min(abs(pr-p))) 
#       qr <- NULL
#       for (j in 2:q1){
#         if (pr[j-1]<p[i] & p[i]<=pr[j]){qr<-j}
#       }
#       if (p[i] == pr[1]){qr <- 2}
#     }
#     else {rq<-NA}
#     r0<-c(r0, qr)
#   }
#   return(r0)  
# }  
# ##
# ##  Generates (pseudo) randon values of the runs statistic
# ##
# rruns <- function(n, n1, n2){
#   return(qruns(runif(n), n1, n2))  
# }    


import jax.numpy as jnp
import jax.tree_util as jtu

from jax import jit, vmap, lax

from ..math import choose
from ..util import zero_dim_to_1_dim_array



@jtu.Partial(jit)
def druns(x, n1, n2):
    cond0 = x == round(x)
    x = jnp.where(cond0, x, 1)
    x = jnp.asarray(x, jnp.int32)
    x = zero_dim_to_1_dim_array(x)
    func1 = lambda xi : 2*choose(n1 - 1, round(xi/ 2) - 1)*choose(n2-1, round(xi/2)-1)
    func2 = lambda xi : choose(n1-1, round((xi-1)/2))*choose(n2-1, round((xi-3)/2))+choose(n1-1, round((xi-3)/2))*choose(n2-1, round((xi-1)/2))

    r0 = vmap(lambda xi: lax.cond(jnp.where(xi // 2 == 0, 1, 0), func1, func2, xi))(x)
    r0 = r0/choose(n1+n2, n1)
    r0 = jnp.squeeze(r0, axis=1)
    return r0

# pruns <- function(q, n1, n2, lower.tail = TRUE, log.p = FALSE){
#   stopifnot(is.numeric(q) & n1>0 & n2>0)
#   q <- ifelse(q >= 1, q, 1)
#   q <- ifelse(q <= n1+n2, q, n1+n2)
#   q <- round(q)
#   tmp <- cumsum(druns(1:max(q),n1,n2,log=log.p))
#   r0 <- tmp[q]
#   if (lower.tail==FALSE){r0<- 1-r0}  
# #  r0 <- NULL
# #  if (lower.tail){
# #    for (i in 1:length(q)){r0 <- c(r0,ifelse(q[i]>=2,sum(druns(x=2:floor(q[i]),n1,n2,log=log)),0))}
# #  }
# #  else {r0 <- 1-pruns(q,n1,n2,lower.tail=T, log=log)}  
#   return(r0)  
# }  
@jtu.Partial(jit, static_argnames=('n1','n2', ))
def _cumm_druns(x, n1 ,n2):
    def cond(carry):
        i, k,  _ = carry
        return i <= k
    def body(carry):
        i, k, ds0 = carry
        ds1 = ds0 + druns(i, n1, n2)
        i = i + 1
        carry = (i, k, ds1)
        return carry
    i = 0
    init = (i,  jnp.max(x),  jnp.asarray([0.]))
    out = lax.while_loop(cond, body, init)
    return out[2] 

def pruns(x, n1, n2):
    x = jnp.where(x >= 1, x, 1)
    x = jnp.where(x <= n1 + n2, x, 1)
    x = jnp.round(x)
    x = jnp.asarray(x, dtype=jnp.int32)
    r0 = _cumm_druns(x, n1, n2)
    return r0

# qruns <- function(p, n1, n2, lower.tail = TRUE, log.p = FALSE){
#   r0 <- NULL
#   q1 <- ifelse (n1==n2, 2*n1, 2*min(n1,n2)+1) 
#   pr <- c(0, cumsum(druns(2:q1, n1, n2)))
#   for (i in 1:length(p)){
#     if (p[i]>=0 & p[i]<=1){
#       #rq<-which(abs(pr-p)==min(abs(pr-p))) 
#       qr <- NULL
#       for (j in 2:q1){
#         if (pr[j-1]<p[i] & p[i]<=pr[j]){qr<-j}
#       }
#       if (p[i] == pr[1]){qr <- 2}
#     }
#     else {rq<-NA}
#     r0<-c(r0, qr)
#   }
#   return(r0)  
# }  
