from typing import Optional

from jaxtyping import Array, Float

from ._fourier import (
    fourier_MAE,
    fourier_MSE,
    fourier_nMAE,
    fourier_nMSE,
    fourier_nRMSE,
    fourier_RMSE,
)


def H1_MAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> float:
    """
    Compute the mean abolute error associated with the H1 norm, i.e., the MAE
    across state and all its first derivatives.

    This is **not** consistent with the H1 norm because it uses a Fourier-based
    approach.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! warning
        Not supplying `domain_extent` will have the result be orders of
        magnitude different.


    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention
        with a leading channel axis, and either one, two, or three subsequent
        spatial axes.
    - `u_ref`: The reference state array. Must have the same shape as
        `u_pred`. If not specified, the MAE is computed against zero, i.e., the
        norm of `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    """
    regular_mae = fourier_MAE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=None,
    )
    first_derivative_mae = fourier_MAE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=1,
    )
    return regular_mae + first_derivative_mae


def H1_nMAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> float:
    """
    Compute the normalized mean abolute error associated with the H1 norm, i.e.,
    the nMAE across state and all its first derivatives.

    This is **not** consistent with the H1 norm because it uses a Fourier-based
    approach.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! warning
        Not supplying `domain_extent` will have the result be orders of
        magnitude different.

    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention
        with a leading channel axis, and either one, two, or three subsequent
        spatial axes.
    - `u_ref`: The reference state array. Must have the same shape as
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    """
    regular_nmae = fourier_nMAE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=None,
    )
    first_derivative_nmae = fourier_nMAE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=1,
    )
    return regular_nmae + first_derivative_nmae


def H1_MSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> float:
    """
    Compute the mean squared error associated with the H1 norm, i.e., the MSE
    across state and all its first derivatives.

    Given the correct `domain_extent`, this is consistent with the squared norm
    in the H1 Sobolev space H^1 = W^(1,2):
    https://en.wikipedia.org/wiki/Sobolev_space#The_case_p_=_2

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! warning
        Not supplying `domain_extent` will have the result be orders of
        magnitude different.

    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention
        with a leading channel axis, and either one, two, or three subsequent
        spatial axes.
    - `u_ref`: The reference state array. Must have the same shape as
        `u_pred`. If not specified, the MSE is computed against zero, i.e., the
        norm of `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    """
    regular_mse = fourier_MSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=None,
    )
    first_derivative_mse = fourier_MSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=1,
    )
    return regular_mse + first_derivative_mse


def H1_nMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> float:
    """
    Compute the normalized mean squared error associated with the H1 norm, i.e.,
    the nMSE across state and all its first derivatives.

    Given the correct `domain_extent`, this is consistent with the **relative**
    squared norm in the H1 Sobolev space H^1 = W^(1,2):
    https://en.wikipedia.org/wiki/Sobolev_space#The_case_p_=_2

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! warning
        Not supplying `domain_extent` will have the result be orders of
        magnitude different.

    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention
        with a leading channel axis, and either one, two, or three subsequent
        spatial axes.
    - `u_ref`: The reference state array. Must have the same shape as
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    """
    regular_nmse = fourier_nMSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=None,
    )
    first_derivative_nmse = fourier_nMSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=1,
    )
    return regular_nmse + first_derivative_nmse


def H1_RMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> float:
    """
    Compute the root mean squared error associated with the H1 norm, i.e., the
    RMSE across state and all its first derivatives.

    Given the correct `domain_extent`, this is consistent with the norm in the
    H1 Sobolev space H^1 = W^(1,2):
    https://en.wikipedia.org/wiki/Sobolev_space#The_case_p_=_2

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! warning
        Not supplying `domain_extent` will have the result be orders of
        magnitude different.

    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention
        with a leading channel axis, and either one, two, or three subsequent
        spatial axes.
    - `u_ref`: The reference state array. Must have the same shape as
        `u_pred`. If not specified, the RMSE is computed against zero, i.e., the
        norm of `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    """
    regular_rmse = fourier_RMSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=None,
    )
    first_derivative_rmse = fourier_RMSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=1,
    )
    return regular_rmse + first_derivative_rmse


def H1_nRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> float:
    """
    Compute the normalized root mean squared error associated with the H1 norm,
    i.e., the nRMSE across state and all its first derivatives.

    Given the correct `domain_extent`, this is consistent with the **relative**
    norm in the H1 Sobolev space H^1 = W^(1,2):
    https://en.wikipedia.org/wiki/Sobolev_space#The_case_p_=_2

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! warning
        Not supplying `domain_extent` will have the result be orders of
        magnitude different.

    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention
        with a leading channel axis, and either one, two, or three subsequent
        spatial axes.
    - `u_ref`: The reference state array. Must have the same shape as
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    """
    regular_nrmse = fourier_nRMSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=None,
    )
    first_derivative_nrmse = fourier_nRMSE(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        low=low,
        high=high,
        derivative_order=1,
    )
    return regular_nrmse + first_derivative_nrmse
