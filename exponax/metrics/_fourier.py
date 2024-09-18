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
    Sums up the gradient contirbutions
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
    """Under normalized mode each channel is normalized separately. The channel is summed"""
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
