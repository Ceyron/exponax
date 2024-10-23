import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _correlation(
    u_pred: Float[Array, "... N"],
    u_ref: Float[Array, "... N"],
) -> float:
    """
    Low-level function to compute the correlation between two fields.

    This function assumes field without channel axes. Even for singleton channel
    axes, use `correlation` for correct operation.

    **Arguments**:

    - `u_pred` (array): The first field to be used in the loss
    - `u_ref` (array): The second field to be used in the error computation

    **Returns**:

    - `correlation` (float): The correlation between the fields
    """
    u_pred_normalized = u_pred / jnp.linalg.norm(u_pred)
    u_ref_normalized = u_ref / jnp.linalg.norm(u_ref)

    correlation = jnp.dot(u_pred_normalized.flatten(), u_ref_normalized.flatten())

    return correlation


def correlation(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
) -> float:
    """
    Compute the correlation between two fields. Average over all channels.

    This function assumes that the arrays have one leading channel axis and an
    arbitrary number of following spatial axes.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    **Arguments**:

    - `u_pred`: The first field to be used in the error computation.
    - `u_ref`: The second field to be used in the error computation.

    **Returns**:

    - `correlation`: The correlation between the fields, averaged over
        all channels.
    """
    channel_wise_correlation = jax.vmap(_correlation)(u_pred, u_ref)
    correlation = jnp.mean(channel_wise_correlation)
    return correlation
