from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


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
    `jax.vmap` on this function or use the [`mean_MSE`](#mean_MSE) function.

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

    In contrast to [`MSE`](#MSE), no `domain_extent` is required, because of the
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
    `jax.vmap` on this function or use the [`mean_RMSE`](#mean_RMSE) function.

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

    In contrast to [`RMSE`](#RMSE), no `domain_extent` is required, because of
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
