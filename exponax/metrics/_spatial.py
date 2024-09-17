from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def spatial_aggregator(
    state_no_channel: Float[Array, "... N"],
    *,
    num_spatial_dims: Optional[int] = None,
    domain_extent: float = 1.0,
    num_points: Optional[int] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
) -> float:
    """
    mean aggregation over space as a consistent counterpart to integrating over
    space in the continuous case
    """
    if num_spatial_dims is None:
        num_spatial_dims = state_no_channel.ndim
    if num_points is None:
        num_points = state_no_channel.shape[-1]

    if outer_exponent is None:
        outer_exponent = 1 / inner_exponent

    scale = (domain_extent / num_points) ** num_spatial_dims

    aggregated = jnp.sum(jnp.abs(state_no_channel) ** inner_exponent)

    return (scale * aggregated) ** outer_exponent


def spatial_norm(
    state: Float[Array, "C ... N"],
    state_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    mode: Literal["absolute", "normalized"] = "absolute",
    domain_extent: float = 1.0,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
) -> float:
    """Under normalized mode each channel is normalized separately. The channel is summed"""
    if state_ref is None:
        if mode == "normalized":
            raise ValueError("mode 'normalized' requires state_ref")
        diff = state
    else:
        diff = state - state_ref

    diff_norm_per_channel = jax.vmap(
        lambda s: spatial_aggregator(
            s,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
        ),
    )(diff)

    if mode == "normalized":
        ref_norm_per_channel = jax.vmap(
            lambda r: spatial_aggregator(
                r,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
            ),
        )(state_ref)
        normalized_diff_per_channel = diff_norm_per_channel / ref_norm_per_channel
        norm_per_channel = normalized_diff_per_channel
    else:
        norm_per_channel = diff_norm_per_channel

    return jnp.sum(norm_per_channel)


def MAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
) -> float:
    return spatial_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
    )


def nMAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    return spatial_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
    )


def MSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
) -> float:
    return spatial_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
    )


def nMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    return spatial_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
    )


def RMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
) -> float:
    return spatial_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=0.5,
    )


def nRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    return spatial_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=0.5,
    )
