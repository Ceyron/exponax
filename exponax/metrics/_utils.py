import jax
import jax.numpy as jnp


def mean_metric(
    metric_fn,
    *args,
    **kwargs,
):
    """
    'meanifies' a metric function to operate on arrays with a leading batch axis
    """
    wrapped_fn = lambda *a: metric_fn(*a, **kwargs)
    metric_per_sample = jax.vmap(wrapped_fn, in_axes=0)(*args)
    return jnp.mean(metric_per_sample, axis=0)
