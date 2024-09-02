from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._spectral import low_pass_filter_mask, space_indices


def _MSE(
    u_pred: Float[Array, "... N"],
    u_ref: Optional[Float[Array, "... N"]] = None,
    domain_extent: float = 1.0,
    *,
    num_spatial_dims: Optional[int] = None,
) -> float:
    """
    Low-level function to compute the mean squared error (MSE) correctly scaled
    for states representing physical fields on uniform Cartesian grids.

    MSE = 1/L^D * 1/N * sum_i (u_pred_i - u_ref_i)^2

    Note that by default (`num_spatial_dims=None`), the number of spatial
    dimensions is inferred from the shape of the input fields. Please adjust
    this argument if you call this function with an array that also contains
    channels (even for arrays with singleton channels.

    Providing correct information regarding the scaling (i.e. providing
    `domain_extent` and `num_spatial_dims`) is not necessary if the result is
    used to compute a normalized error (e.g. nMSE) if the normalization is
    computed similarly.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the loss
        - `u_ref` (array, optional): The second field to be used in the error
            computation. If `None`, the error will be computed with respect to
            zero.
        - `domain_extent` (float, optional): The extent of the domain in which
            the fields are defined. This is used to scale the error to be
            independent of the domain size. Default is 1.0.
        - `num_spatial_dims` (int, optional): The number of spatial dimensions
            in the field. If `None`, it will be inferred from the shape of the
            input fields and then is the number of axes present. Default is
            `None`.

    **Returns**:
        - `mse` (float): The (correctly scaled) mean squared error between the
          fields.
    """
    if u_ref is None:
        diff = u_pred
    else:
        diff = u_pred - u_ref

    if num_spatial_dims is None:
        # Assuming that we only have spatial dimensions
        num_spatial_dims = len(u_pred.shape)

    scale = 1 / (domain_extent**num_spatial_dims)

    mse = scale * jnp.mean(jnp.square(diff))

    return mse


def MSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    domain_extent: float = 1.0,
):
    """
    Compute the mean squared error (MSE) between two fields.

    This function assumes that the arrays have one leading channel axis and an
    arbitrary number of following spatial dimensions! For batched operation use
    `jax.vmap` on this function or use the [`exponax.metrics.mean_MSE`][] function.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array, optional): The second field to be used in the error
            computation. If `None`, the error will be computed with respect to
            zero.
        - `domain_extent` (float, optional): The extent of the domain in which
            the fields are defined. This is used to scale the error to be
            independent of the domain size. Default is 1.0.

    **Returns**:
        - `mse` (float): The (correctly scaled) mean squared error between the
            fields.
    """

    num_spatial_dims = len(u_pred.shape) - 1

    mse = _MSE(u_pred, u_ref, domain_extent, num_spatial_dims=num_spatial_dims)

    return mse


def nMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
) -> float:
    """
    Compute the normalized mean squared error (nMSE) between two fields.

    In contrast to [`exponax.metrics.MSE`][], no `domain_extent` is required, because of the
    normalization.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array): The second field to be used in the error computation.
            This is also used to normalize the error.

    **Returns**:
        - `nmse` (float): The normalized mean squared error between the fields
    """

    num_spatial_dims = len(u_pred.shape) - 1

    # Do not have to supply the domain_extent, because we will normalize with
    # the ref_mse
    diff_mse = _MSE(u_pred, u_ref, num_spatial_dims=num_spatial_dims)
    ref_mse = _MSE(u_ref, num_spatial_dims=num_spatial_dims)

    nmse = diff_mse / ref_mse

    return nmse


def mean_MSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the mean MSE between two fields. Use this function to correctly
    operate on arrays with a batch axis.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array): The second field to be used in the error computation.
        - `domain_extent` (float, optional): The extent of the domain in which
            the fields are defined. This is used to scale the error to be
            independent of the domain size. Default is 1.0.

    **Returns**:
        - `mean_mse` (float): The mean mean squared error between the fields
    """
    batch_wise_mse = jax.vmap(MSE, in_axes=(0, 0, None))(u_pred, u_ref, domain_extent)
    mean_mse = jnp.mean(batch_wise_mse)
    return mean_mse


def mean_nMSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
):
    """
    Compute the mean nMSE between two fields. Use this function to correctly
    operate on arrays with a batch axis.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array): The second field to be used in the error computation.

    **Returns**:
        - `mean_nmse` (float): The mean normalized mean squared error between
    """
    batch_wise_nmse = jax.vmap(nMSE)(u_pred, u_ref)
    mean_nmse = jnp.mean(batch_wise_nmse)
    return mean_nmse


def _RMSE(
    u_pred: Float[Array, "... N"],
    u_ref: Optional[Float[Array, "... N"]] = None,
    domain_extent: float = 1.0,
    *,
    num_spatial_dims: Optional[int] = None,
) -> float:
    """
    Low-level function to compute the root mean squared error (RMSE) correctly
    scaled for states representing physical fields on uniform Cartesian grids.

    RMSE = sqrt(1/L^D * 1/N * sum_i (u_pred_i - u_ref_i)^2)

    Note that by default (`num_spatial_dims=None`), the number of spatial
    dimensions is inferred from the shape of the input fields. Please adjust
    this argument if you call this function with an array that also contains
    channels (even for arrays with singleton channels!).

    Providing correct information regarding the scaling (i.e. providing
    `domain_extent` and `num_spatial_dims`) is not necessary if the result is
    used to compute a normalized error (e.g. nRMSE) if the normalization is
    computed similarly.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the loss
        - `u_ref` (array, optional): The second field to be used in the error
            computation. If `None`, the error will be computed with respect to
            zero.
        - `domain_extent` (float, optional): The extent of the domain in which
            the fields are defined. This is used to scale the error to be
            independent of the domain size. Default is 1.0.
        - `num_spatial_dims` (int, optional): The number of spatial dimensions
            in the field. If `None`, it will be inferred from the shape of the
            input fields and then is the number of axes present. Default is
            `None`.

    **Returns**:
        - `rmse` (float): The (correctly scaled) root mean squared error between
          the fields.
    """
    if u_ref is None:
        diff = u_pred
    else:
        diff = u_pred - u_ref

    if num_spatial_dims is None:
        # Assuming that we only have spatial dimensions
        num_spatial_dims = len(u_pred.shape)

    # Todo: Check if we have to divide by 1/L or by 1/L^D for D dimensions
    scale = 1 / (domain_extent**num_spatial_dims)

    rmse = jnp.sqrt(scale * jnp.mean(jnp.square(diff)))
    return rmse


def RMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the root mean squared error (RMSE) between two fields.

    This function assumes that the arrays have one leading channel axis and an
    arbitrary number of following spatial dimensions! For batched operation use
    `jax.vmap` on this function or use the [`exponax.metrics.mean_RMSE`][] function.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array, optional): The second field to be used in the error
            computation. If `None`, the error will be computed with respect to
            zero.
        - `domain_extent` (float, optional): The extent of the domain in which
            the fields are defined. This is used to scale the error to be
            independent of the domain size. Default is 1.0.

    **Returns**:
        - `rmse` (float): The (correctly scaled) root mean squared error between
            the fields.
    """

    num_spatial_dims = len(u_pred.shape) - 1

    rmse = _RMSE(u_pred, u_ref, domain_extent, num_spatial_dims=num_spatial_dims)

    return rmse


def nRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
) -> float:
    """
    Compute the normalized root mean squared error (nRMSE) between two fields.

    In contrast to [`exponax.metrics.RMSE`][], no `domain_extent` is required, because of
    the normalization.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array): The second field to be used in the error computation.

    **Returns**:
        - `nrmse` (float): The normalized root mean squared error between the
            fields
    """

    num_spatial_dims = len(u_pred.shape) - 1

    # Do not have to supply the domain_extent, because we will normalize with
    # the ref_rmse
    diff_rmse = _RMSE(u_pred, u_ref, num_spatial_dims=num_spatial_dims)
    ref_rmse = _RMSE(u_ref, num_spatial_dims=num_spatial_dims)

    nrmse = diff_rmse / ref_rmse

    return nrmse


def mean_RMSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the mean RMSE between two fields. Use this function to correctly
    operate on arrays with a batch axis.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array): The second field to be used in the error computation.
        - `domain_extent` (float, optional): The extent of the domain in which

    **Returns**:
        - `mean_rmse` (float): The mean root mean squared error between the
            fields
    """
    batch_wise_rmse = jax.vmap(RMSE, in_axes=(0, 0, None))(u_pred, u_ref, domain_extent)
    mean_rmse = jnp.mean(batch_wise_rmse)
    return mean_rmse


def mean_nRMSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
):
    """
    Compute the mean nRMSE between two fields. Use this function to correctly
    operate on arrays with a batch axis.

    **Arguments**:
        - `u_pred` (array): The first field to be used in the error computation.
        - `u_ref` (array): The second field to be used in the error computation.

    **Returns**:
        - `mean_nrmse` (float): The mean normalized root mean squared error
    """
    batch_wise_nrmse = jax.vmap(nRMSE)(u_pred, u_ref)
    mean_nrmse = jnp.mean(batch_wise_nrmse)
    return mean_nrmse


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


# # Below seems to produce the same resuls as `correlation`
# def pearson_correlation(
#     u_pred: Float[Array, "... N"],
#     u_ref: Float[Array, "... N"],
# ) -> float:
#     """
#     Based on
#     https://github.com/pdearena/pdearena/blob/22360a766387c3995220b4a1265a936ab9a81b88/pdearena/modules/loss.py#L39
#     """

#     u_pred_mean = jnp.mean(u_pred)
#     u_ref_mean = jnp.mean(u_ref)

#     u_pred_centered = u_pred - u_pred_mean
#     u_ref_centered = u_ref - u_ref_mean

#     # u_pred_std = jnp.sqrt(jnp.mean(u_pred_centered ** 2))
#     # u_ref_std = jnp.sqrt(jnp.mean(u_ref_centered ** 2))

#     u_pred_std = jnp.std(u_pred)
#     u_ref_std = jnp.std(u_ref)

#     # numerator = jnp.sum(u_pred_centered * u_ref_centered)
#     # denominator = jnp.sqrt(jnp.sum(u_pred_centered ** 2) * jnp.sum(u_ref_centered ** 2))

#     # correlation = numerator / denominator

#     correlation = jnp.mean(u_pred_centered * u_ref_centered) / (u_pred_std * u_ref_std)

#     return correlation


def _fourier_nRMSE(
    u_pred: Float[Array, "... N"],
    u_ref: Float[Array, "... N"],
    *,
    low: Optional[int] = None,
    high: Optional[int] = None,
    num_spatial_dims: Optional[int] = None,
    eps: float = 1e-5,
) -> float:
    """
    Low-level function to compute the normalized root mean squared error (nRMSE)
    between two fields in Fourier space.

    If `num_spatial_dims` is not provided, it will be inferred from the shape of
    the input fields. Please adjust this argument if you call this function with
    an array that also contains channels (even for arrays with singleton
    channels).

    **Arguments**:

    - `u_pred` (array): The first field to be used in the error computation.
    - `u_ref` (array): The second field to be used in the error computation.
    - `low` (int, optional): The low-pass filter cutoff. Default is 0.
    - `high` (int, optional): The high-pass filter cutoff. Default is the
        Nyquist frequency.
    - `num_spatial_dims` (int, optional): The number of spatial dimensions in
        the field. If `None`, it will be inferred from the shape of the input
        fields and then is the number of axes present. Default is `None`.
    - `eps` (float, optional): Small value to avoid division by zero and to
        remove numerical rounding artiacts from the FFT. Default is 1e-5.
    """
    if num_spatial_dims is None:
        num_spatial_dims = len(u_pred.shape)
    # Assumes we have the same N for all dimensions
    num_points = u_pred.shape[-1]

    if low is None:
        low = 0
    if high is None:
        high = (num_points // 2) + 1

    low_mask = low_pass_filter_mask(
        num_spatial_dims,
        num_points,
        cutoff=low - 1,  # Need to subtract 1 because the cutoff is inclusive
    )
    high_mask = low_pass_filter_mask(
        num_spatial_dims,
        num_points,
        cutoff=high,
    )

    mask = jnp.invert(low_mask) & high_mask

    u_pred_fft = jnp.fft.rfftn(u_pred, axes=space_indices(num_spatial_dims))
    u_ref_fft = jnp.fft.rfftn(u_ref, axes=space_indices(num_spatial_dims))

    # The FFT incurse rounding errors around the machine precision that can be
    # noticeable in the nRMSE. We will zero out the values that are smaller than
    # the epsilon to avoid this.
    u_pred_fft = jnp.where(
        jnp.abs(u_pred_fft) < eps,
        jnp.zeros_like(u_pred_fft),
        u_pred_fft,
    )
    u_ref_fft = jnp.where(
        jnp.abs(u_ref_fft) < eps,
        jnp.zeros_like(u_ref_fft),
        u_ref_fft,
    )

    u_pred_fft_masked = u_pred_fft * mask
    u_ref_fft_masked = u_ref_fft * mask

    diff_fft_masked = u_pred_fft_masked - u_ref_fft_masked

    # Need to use vdot to correctly operate with complex numbers
    diff_norm_unscaled = jnp.sqrt(
        jnp.vdot(diff_fft_masked.flatten(), diff_fft_masked.flatten())
    ).real
    ref_norm_unscaled = jnp.sqrt(
        jnp.vdot(u_ref_fft_masked.flatten(), u_ref_fft_masked.flatten())
    ).real

    nrmse = diff_norm_unscaled / (ref_norm_unscaled + eps)

    return nrmse


def fourier_nRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    low: Optional[int] = None,
    high: Optional[int] = None,
    eps: float = 1e-5,
) -> float:
    """
    Compute the normalized root mean squared error (nRMSE) between two fields
    in Fourier space.

    **Arguments**:

    - `u_pred` (array): The first field to be used in the error computation.
    - `u_ref` (array): The second field to be used in the error computation.
    - `low` (int, optional): The low-pass filter cutoff. Default is 0.
    - `high` (int, optional): The high-pass filter cutoff. Default is the Nyquist
        frequency.
    - `eps` (float, optional): Small value to avoid division by zero and to
        remove numerical rounding artiacts from the FFT. Default is 1e-5.

    **Returns**:

    - `nrmse` (float): The normalized root mean squared error between the fields
    """
    num_spatial_dims = len(u_pred.shape) - 1

    nrmse = _fourier_nRMSE(
        u_pred, u_ref, low=low, high=high, num_spatial_dims=num_spatial_dims, eps=eps
    )

    return nrmse


def mean_fourier_nRMSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    *,
    low: Optional[int] = None,
    high: Optional[int] = None,
    eps: float = 1e-5,
) -> float:
    """
    Compute the mean nRMSE between two fields in Fourier space. Use this function
    to correctly operate on arrays with a batch axis.

    **Arguments**:

    - `u_pred` (array): The first field to be used in the error computation.
    - `u_ref` (array): The second field to be used in the error computation.
    - `low` (int, optional): The low-pass filter cutoff. Default is 0.
    - `high` (int, optional): The high-pass filter cutoff. Default is the Nyquist
        frequency.
    - `eps` (float, optional): Small value to avoid division by zero and to
        remove numerical rounding artiacts from the FFT. Default is 1e-5.

    **Returns**:

    - `mean_nrmse` (float): The mean normalized root mean squared error between the
        fields
    """
    batch_wise_nrmse = jax.vmap(
        lambda pred, ref: fourier_nRMSE(pred, ref, low=low, high=high, eps=eps)
    )(u_pred, u_ref)
    mean_nrmse = jnp.mean(batch_wise_nrmse)
    return mean_nrmse
