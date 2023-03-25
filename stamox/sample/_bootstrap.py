from jaxtyping import ArrayLike
import jax.random as jrandom
from equinox import filter_vmap, filter_jit

from ..core import make_partial_pipe

@make_partial_pipe
def bootstrap_sample(data: ArrayLike, num_samples: int, *, key: jrandom.KeyArray = None):
    """
    Generate `num_samples` bootstrap samples from `data` with replacement.

    Parameters:
        data (array-like): The original data.
        num_samples (int): The number of bootstrap samples to generate.

    Returns:
        (numpy.ndarray): An array of size (num_samples, len(data))
                          representing the bootstrap samples.
    """
    # Determine the number of elements in the data
    n = data.shape[0]
    keys = jrandom.split(key, num_samples)

    # @filter_jit
    def sample_fn(key: jrandom.KeyArray):
        # Draw n random indices from the data, with replacement
        sample_indices = jrandom.choice(key, n, (n, ), replace=True)
        # Use the drawn indices to create a new bootstrap sample
        sample = data[sample_indices,:]
        # Add the bootstrap sample to the samples array
        return sample
    
    samples = filter_vmap(sample_fn)(keys)
    return samples


