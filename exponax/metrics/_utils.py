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
    return jnp.mean(jax.vmap(wrapped_fn)(*args))
