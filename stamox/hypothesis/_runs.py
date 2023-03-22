# druns <- function(x, n1, n2, log = FALSE){
#   stopifnot(is.numeric(x))
#   x <- ifelse(x == round(x),x,1)
#   r0 <- ifelse(x %% 2==0, 2*choose(n1-1, round(x/2)-1)*choose(n2-1, round(x/2)-1), 
#              choose(n1-1, round((x-1)/2))*choose(n2-1, round((x-3)/2))+choose(n1-1, round((x-3)/2))*choose(n2-1, round((x-1)/2)))  
#   r0<-r0/choose(n1+n2, n1)
# # if TRUE, probabilities p are given as log(p).  
# ifelse(log,return(log(r0)),return(r0))  
# }

import jax.numpy as jnp
from jax import jit, vmap, lax

from ..math import choose



