from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .._spectral import (
    build_derivative_operator,
    build_scaling_array,
    fft,
    low_pass_filter_mask,
)


def fourier_aggregator(
    state_no_channel: Float[Array, "... N"],
    *,
    num_spatial_dims: Optional[int] = None,
    domain_extent: float = 1.0,
    num_points: Optional[int] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Aggregate over the spatial axes of a (channel-less) state array in Fourier
    space.

    While conceptually similar to [`exponax.metrics.spatial_aggregator`][], this
    function additionally allows filtering specific frequency ranges and to take
    derivatives. In higher dimensions, the derivative contributions (i.e., the
    entries of the gradient) are summed up.

    !!! info
        The result of this function (under default settings) is (up to rounding
        errors) identical to [`exponax.metrics.spatial_aggregator`][] for
        `inner_exponent=2.0`. As such, it can be a consistent counterpart for
        metrics based on the `L²(Ω)` functional norm.

    !!! tip
        To apply this function to a state tensor with a leading channel axis,
        use `jax.vmap`.

    **Arguments**:

    - `state_no_channel`: The state tensor **without a leading channel
        dimension**.
    - `num_spatial_dims`: The number of spatial dimensions. If not specified,
        it is inferred from the number of axes in `state_no_channel`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `num_points`: The number of points `N` in each spatial dimension. If not
        specified, it is inferred from the last axis of `state_no_channel`.
    - `inner_exponent`: The exponent `p` each magnitude of a Fourier coefficient
        is raised to before aggregation.
    - `outer_exponent`: The exponent `q` the aggregated magnitudes are raised
        to. If not specified, it is set to `1/p`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    if num_spatial_dims is None:
        num_spatial_dims = state_no_channel.ndim
    if num_points is None:
        num_points = state_no_channel.shape[-1]

    if outer_exponent is None:
        outer_exponent = 1 / inner_exponent

    # Transform to Fourier space
    state_no_channel_hat = fft(state_no_channel, num_spatial_dims=num_spatial_dims)

    # Remove small values that occured due to rounding errors, can become
    # problematic for "normalized" norms
    state_no_channel_hat = jnp.where(
        jnp.abs(state_no_channel_hat) < 1e-5,
        jnp.zeros_like(state_no_channel_hat),
        state_no_channel_hat,
    )

    # Filtering out if desired
    if low is not None or high is not None:
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

        state_no_channel_hat = state_no_channel_hat * mask

    # Taking derivatives if desired
    if derivative_order is not None:
        derivative_operator = build_derivative_operator(
            num_spatial_dims, domain_extent, num_points
        )
        state_with_derivative_channel_hat = (
            state_no_channel_hat * derivative_operator**derivative_order
        )
    else:
        # Add singleton derivative axis to have subsequent code work
        state_with_derivative_channel_hat = state_no_channel_hat[None]

    # Scale coefficients to extract the correct form, this is needed because we
    # use the rfft
    scaling_array_recon = build_scaling_array(
        num_spatial_dims,
        num_points,
        mode="reconstruction",
    )

    scale = (domain_extent / num_points) ** num_spatial_dims

    def aggregate(s):
        scaled_coefficient_magnitude = (
            jnp.abs(s) ** inner_exponent / scaling_array_recon
        )
        aggregated = jnp.sum(scaled_coefficient_magnitude)
        return (scale * aggregated) ** outer_exponent

    aggregated_per_derivative = jax.vmap(aggregate)(state_with_derivative_channel_hat)

    return jnp.sum(aggregated_per_derivative)


def fourier_norm(
    state: Float[Array, "C ... N"],
    state_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    mode: Literal["absolute", "normalized"] = "absolute",
    domain_extent: float = 1.0,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Compute norms of states via aggregation in Fourier space.

    Each channel is treated separately and the results are summed up.

    While conceptually similar to [`exponax.metrics.spatial_norm`][], this
    function additionally allows filtering specific frequency ranges and to take
    derivatives. In higher dimensions, the derivative contributions (i.e., the
    entries of the gradient) are summed up.

    !!! tip
        To operate on states with a leading batch axis, use `jax.vmap`. Then the
        batch axis can be reduced, e.g., by `jnp.mean`. As a helper for this,
        [`exponax.metrics.mean_metric`][] is provided.

    If both `low` and `high` are `None`, the full spectrum is considered. In
    this case, this function with `inner_exponent=2.0` (up to rounding errors)
    produces the same result as [`exponax.metrics.spatial_norm`][] which is a
    consequence of Parseval's theorem.


    **Arguments**:

    - `state`: The state tensor. Must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `state_ref`: The reference state tensor. Must have the same shape as
        `state`. If not specified, only the absolute norm of `state` is
        computed.
    - `mode`: The mode of the norm. Either `"absolute"` or `"normalized"`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `inner_exponent`: The exponent `p` each magnitude of a Fourier coefficient
        is raised to before aggregation.
    - `outer_exponent`: The exponent `q` the aggregated magnitudes are raised
        to. If not specified, it is set to `1/p`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    if state_ref is None:
        if mode == "normalized":
            raise ValueError("mode 'normalized' requires state_ref")
        diff = state
    else:
        diff = state - state_ref

    diff_norm_per_channel = jax.vmap(
        lambda s: fourier_aggregator(
            s,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
            low=low,
            high=high,
            derivative_order=derivative_order,
        ),
    )(diff)

    if mode == "normalized":
        ref_norm_per_channel = jax.vmap(
            lambda r: fourier_aggregator(
                r,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
                low=low,
                high=high,
                derivative_order=derivative_order,
            ),
        )(state_ref)
        normalized_diff_per_channel = diff_norm_per_channel / ref_norm_per_channel
        norm_per_channel = normalized_diff_per_channel
    else:
        norm_per_channel = diff_norm_per_channel

    return jnp.sum(norm_per_channel)


def fourier_MAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Compute the mean absolute error in Fourier space.

        ∑_(channels) ∑_(modi) (L/N)ᴰ |fft(uₕ - uₕʳ)|

    The channel axis is summed **after** the aggregation.

    While conceptually similar to [`exponax.metrics.MAE`][], this
    function is **not** consistent with the `L¹(Ω)` functional norm. However, it
    additionally allows filtering specific frequency ranges and to take
    derivatives. In higher dimensions, the derivative contributions (i.e., the
    entries of the gradient) are summed up.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


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
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    return fourier_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
        low=low,
        high=high,
        derivative_order=derivative_order,
    )


def fourier_nMAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Compute the normalized mean absolute error in Fourier space.

        ∑_(channels) (∑_(modi) (L/N)ᴰ |fft(uₕ - uₕʳ)| / ∑_(modi) (L/N)ᴰ
        |fft(uₕʳ)|)

    The channel axis is summed **after** the aggregation.

    While conceptually similar to [`exponax.metrics.nMAE`][], this
    function is **not** consistent with the `L¹(Ω)` functional norm. However, it
    additionally allows filtering specific frequency ranges and to take
    derivatives. In higher dimensions, the derivative contributions (i.e., the
    entries of the gradient) are summed up.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    return fourier_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
        low=low,
        high=high,
        derivative_order=derivative_order,
    )


def fourier_MSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Compute the mean squared error in Fourier space.

        ∑_(channels) ∑_(modi) (L/N)ᴰ |fft(uₕ - uₕʳ)|²

    The channel axis is summed **after** the aggregation.

    Under default settings with correctly specific `domain_extent`, this
    function (up to rounding errors) produces the identical result as
    [`exponax.metrics.MSE`][] which is a consequence of Parseval's theorem.
    However, it additionally allows filtering specific frequency ranges and to
    take derivatives. In higher dimensions, the derivative contributions (i.e.,
    the entries of the gradient) are summed up.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
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
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    return fourier_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
        low=low,
        high=high,
        derivative_order=derivative_order,
    )


def fourier_nMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Compute the normalized mean squared error in Fourier space.

        ∑_(channels) (∑_(modi) (L/N)ᴰ |fft(uₕ - uₕʳ)|² / ∑_(modi) (L/N)ᴰ
        |fft(uₕʳ)|²)

    The channel axis is summed **after** the aggregation.

    Under default settings with correctly specific `domain_extent`, this
    function (up to rounding errors) produces the identical result as
    [`exponax.metrics.nMSE`][] which is a consequence of Parseval's theorem.
    However, it additionally allows filtering specific frequency ranges and to
    take derivatives. In higher dimensions, the derivative contributions (i.e.,
    the entries of the gradient) are summed up.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    **Arguments:**

    - `u_pred`: The state array. Must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    return fourier_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
        low=low,
        high=high,
        derivative_order=derivative_order,
    )


def fourier_RMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Compute the root mean squared error in Fourier space.

        ∑_(channels) √(∑_(modi) (L/N)ᴰ |fft(uₕ - uₕʳ)|²)

    The channel axis is summed **after** the aggregation.

    Under default settings with correctly specific `domain_extent`, this
    function (up to rounding errors) produces the identical result as
    [`exponax.metrics.RMSE`][] which is a consequence of Parseval's theorem.
    However, it additionally allows filtering specific frequency ranges and to
    take derivatives. In higher dimensions, the derivative contributions (i.e.,
    the entries of the gradient) are summed up.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
        If not specified, the RMSE is computed against zero, i.e., the norm of
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    return fourier_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=0.5,
        low=low,
        high=high,
        derivative_order=derivative_order,
    )


def fourier_nRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
) -> float:
    """
    Compute the normalized root mean squared error in Fourier space.

        ∑_(channels) (√(∑_(modi) (L/N)ᴰ |fft(uₕ - uₕʳ)|²) / √(∑_(modi) (L/N)ᴰ
        |fft(uₕʳ)|²))

    The channel axis is summed **after** the aggregation.

    Under default settings with correctly specific `domain_extent`, this
    function (up to rounding errors) produces the identical result as
    [`exponax.metrics.nRMSE`][] which is a consequence of Parseval's theorem.
    However, it additionally allows filtering specific frequency ranges and to
    take derivatives. In higher dimensions, the derivative contributions (i.e.,
    the entries of the gradient) are summed up.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    **Arguments**:

    - `u_pred`: The state array. Must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `low`: The lower cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `0`, meaning start it starts (including) the
        mean/zero mode.
    - `high`: The upper cutoff (inclusive) frequency for filtering. If not
        specified, it is set to `N//2 + 1`, meaning it ends (including) at the
        Nyquist mode.
    - `derivative_order`: The order of the derivative to take. If not specified,
        no derivative is taken.
    """
    return fourier_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=0.5,
        low=low,
        high=high,
        derivative_order=derivative_order,
    )
