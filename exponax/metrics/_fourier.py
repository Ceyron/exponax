from typing import Literal, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._spectral import (
    build_derivative_operator,
    build_scaling_array,
    fft,
    low_pass_filter_mask,
)

N = TypeVar("N")
C = TypeVar("C")
D = TypeVar("D")


def _fourier_aggregator_hat(
    state_no_channel_hat: Complex[Array, "... (N//2)+1"],
    *,
    num_spatial_dims: Optional[int] = None,
    num_points: Optional[int] = None,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
    derivative_channel_handling: Literal["mean", "sum", None] = "sum",
    scaling_mode: Literal[
        "norm_compensation", "reconstruction", "coef_extraction", None
    ] = "reconstruction",
) -> Union[float, Float[Array, "D"]]:
    if num_spatial_dims is None:
        num_spatial_dims = state_no_channel_hat.ndim
    if num_points is None:
        if num_spatial_dims >= 2:
            num_points = state_no_channel_hat.shape[-2]
        else:
            raise ValueError("num_points must be provided for 1D")
    if domain_extent is None:
        domain_extent = 1.0

    if outer_exponent is None:
        outer_exponent = 1 / inner_exponent

    # Filtering out if desired
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
        state_no_channel_hat *= derivative_operator**derivative_order
    else:
        # Add singleton derivative axis to have subsequent code work
        state_no_channel_hat = state_no_channel_hat[None]

    # Scale coefficients
    if scaling_mode is not None:
        scaling_array = build_scaling_array(
            num_spatial_dims, num_points, mode=scaling_mode
        )
        state_no_channel_hat /= scaling_array

    def aggregate(s):
        return jnp.sum(jnp.abs(s) ** inner_exponent) ** outer_exponent

    aggregated = jax.vmap(aggregate)(state_no_channel_hat)

    if derivative_channel_handling == "mean":
        return jnp.mean(aggregated)
    elif derivative_channel_handling == "sum":
        return jnp.sum(aggregated)
    else:
        return aggregated


def fourier_aggregator(
    state: Float[Array, "C ... N"],
    *,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
    derivative_channel_handling: Literal["mean", "sum", None] = "sum",
    scaling_mode: Literal[
        "norm_compensation", "reconstruction", "coef_extraction", None
    ] = "reconstruction",
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "D"], Float[Array, "C"], Float[Array, "C D"]]:
    num_spatial_dims = state.ndim - 1
    num_points = state.shape[-1]
    state_hat = fft(state, num_spatial_dims=num_spatial_dims)

    if channel_handling == "spatial":
        return _fourier_aggregator_hat(
            state_hat,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
            low=low,
            high=high,
            derivative_order=derivative_order,
            derivative_channel_handling=derivative_channel_handling,
            scaling_mode=scaling_mode,
        )
    elif channel_handling is None:
        aggregated_per_channel = jax.vmap(
            lambda s: _fourier_aggregator_hat(
                s,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
                low=low,
                high=high,
                derivative_order=derivative_order,
                derivative_channel_handling=derivative_channel_handling,
                scaling_mode=scaling_mode,
            )
        )(state_hat)
        return aggregated_per_channel
    elif channel_handling in ["norm_before", "norm_after"]:
        raise NotImplementedError("This channel handling is not implemented yet")
    else:
        raise ValueError(f"Invalid channel_handling: {channel_handling}")


def fourier_aggregator_diff(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
    derivative_channel_handling: Literal["mean", "sum", None] = "sum",
    scaling_mode: Literal[
        "norm_compensation", "reconstruction", "coef_extraction", None
    ] = "reconstruction",
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "D"], Float[Array, "C"], Float[Array, "C D"]]:
    if u_ref is None:
        diff = u_pred
    else:
        diff = u_pred - u_ref

    return fourier_aggregator(
        diff,
        domain_extent=domain_extent,
        inner_exponent=inner_exponent,
        outer_exponent=outer_exponent,
        low=low,
        high=high,
        derivative_order=derivative_order,
        derivative_channel_handling=derivative_channel_handling,
        scaling_mode=scaling_mode,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def fourier_aggregator_normalized(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
    derivative_channel_handling: Literal["mean", "sum", None] = "sum",
    scaling_mode: Literal[
        "norm_compensation", "reconstruction", "coef_extraction", None
    ] = "reconstruction",
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
):
    pass


def fourier_MSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: Optional[float] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    derivative_order: Optional[float] = None,
    derivative_channel_handling: Literal["mean", "sum", None] = "sum",
    scaling_mode: Literal[
        "norm_compensation", "reconstruction", "coef_extraction", None
    ] = "reconstruction",
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
):
    return fourier_aggregator_diff(
        u_pred,
        u_ref,
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
        low=low,
        high=high,
        derivative_order=derivative_order,
        derivative_channel_handling=derivative_channel_handling,
        scaling_mode=scaling_mode,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )
