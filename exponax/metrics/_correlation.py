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
    arbitrary number of following spatial axes. For operation on batched arrays
    use `mean_correlation`.

    **Arguments**:

    - `u_pred` (array): The first field to be used in the error computation.
    - `u_ref` (array): The second field to be used in the error computation.

    **Returns**:

    - `correlation` (float): The correlation between the fields, averaged over
        all channels.
    """
    channel_wise_correlation = jax.vmap(_correlation)(u_pred, u_ref)
    correlation = jnp.mean(channel_wise_correlation)
    return correlation


def mean_correlation(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
) -> float:
    """
    Compute the mean correlation between multiple samples of two fields.

    This function assumes that the arrays have one leading batch axis, followed
    by a channel axis and an arbitrary number of following spatial axes.

    If you want to apply this function on two trajectories of fields, you can
    use `jax.vmap` to transform it, use `jax.vmap(mean_correlation, in_axes=I)`
    with `I` being the index of the time axis (e.g. `I=0` for time axis at the
    beginning of the array, or `I=1` for time axis at the second position,
    depending on the convention).

    **Arguments**:

    - `u_pred` (array): The first tensor of fields to be used in the error
        computation.
    - `u_ref` (array): The second tensor of fields to be used in the error
        computation.

    **Returns**:

    - `mean_correlation` (float): The mean correlation between the fields
    """
    batch_wise_correlation = jax.vmap(correlation)(u_pred, u_ref)
    mean_correlation = jnp.mean(batch_wise_correlation)
    return mean_correlation
