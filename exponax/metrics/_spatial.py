from typing import Literal, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

C = TypeVar("C")


def _spatial_aggregator(
    state_no_channel: Float[Array, "... N"],
    *,
    num_spatial_dims: Optional[int] = None,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
) -> float:
    if num_spatial_dims is None:
        num_spatial_dims = state_no_channel.ndim
    if domain_extent is None:
        domain_extent = 1.0

    if outer_exponent is None:
        outer_exponent = 1 / inner_exponent

    scale = domain_extent**num_spatial_dims

    aggregated = jnp.mean(jnp.abs(state_no_channel) ** inner_exponent)

    return (scale * aggregated) ** outer_exponent


def spatial_aggregator(
    state: Float[Array, "C ... N"],
    *,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    num_spatial_dims = state.ndim - 1

    if channel_handling == "spatial":
        return _spatial_aggregator(
            state_no_channel=state,
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
        )
    elif channel_handling == "norm_before":
        state_channel_normed = jnp.linalg.norm(
            state, axis=0, ord=channel_handling_norm, keepdims=True
        )
        return _spatial_aggregator(
            state_no_channel=state_channel_normed,
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
        )
    elif channel_handling in ["norm_after", None]:
        aggregated_per_channel = jax.vmap(
            lambda s: _spatial_aggregator(
                state_no_channel=s,
                num_spatial_dims=num_spatial_dims,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
            ),
        )(state)
        if channel_handling == "norm_after":
            return jnp.linalg.norm(aggregated_per_channel, ord=channel_handling_norm)
        else:
            return aggregated_per_channel
    else:
        raise ValueError("Invalid channel_handling value")


def spatial_aggregator_diff(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
):
    if u_ref is None:
        diff = u_pred
    else:
        diff = u_pred - u_ref

    return spatial_aggregator(
        diff,
        domain_extent=domain_extent,
        inner_exponent=inner_exponent,
        outer_exponent=outer_exponent,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def spatial_aggregator_normalized(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: Optional[float] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
    channel_handling: Literal[
        "spatial", "norm_before", "norm_after", "norm_separate", None
    ] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    if channel_handling in ["spatial", "norm_before", "norm_after"]:
        diff_norm = spatial_aggregator_diff(
            u_pred=u_pred,
            u_ref=u_ref,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        )
        ref_norm = spatial_aggregator(
            u_ref,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        )
        return diff_norm / ref_norm
    elif channel_handling in ["norm_separate", None]:
        diff_norm_per_channel = jax.vmap(
            lambda p, r: spatial_aggregator_diff(
                u_pred=p,
                u_ref=r,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
                channel_handling=None,
                channel_handling_norm=None,
            ),
        )(u_pred, u_ref)
        ref_norm_per_channel = jax.vmap(
            lambda r: spatial_aggregator(
                r,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
                channel_handling=None,
                channel_handling_norm=None,
            ),
        )(u_ref)
        normalized_per_channel = diff_norm_per_channel / ref_norm_per_channel
        if channel_handling == "norm_separate":
            return jnp.linalg.norm(normalized_per_channel, ord=channel_handling_norm)
        else:
            return normalized_per_channel
    else:
        raise ValueError("Invalid channel_handling value")


def MSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    return spatial_aggregator_diff(
        u_pred=u_pred,
        u_ref=u_ref,
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def MAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 1,
) -> Union[float, Float[Array, "C"]]:
    return spatial_aggregator_diff(
        u_pred=u_pred,
        u_ref=u_ref,
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def RMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    return spatial_aggregator_diff(
        u_pred=u_pred,
        u_ref=u_ref,
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0 / 2.0,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def nMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal[
        "spatial", "norm_before", "norm_after", "norm_separate", None
    ] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    return spatial_aggregator_normalized(
        u_pred=u_pred,
        u_ref=u_ref,
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def nMAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal[
        "spatial", "norm_before", "norm_after", "norm_separate", None
    ] = "spatial",
    channel_handling_norm: int = 1,
) -> Union[float, Float[Array, "C"]]:
    return spatial_aggregator_normalized(
        u_pred=u_pred,
        u_ref=u_ref,
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def nRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal[
        "spatial", "norm_before", "norm_after", "norm_separate", None
    ] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    return spatial_aggregator_normalized(
        u_pred=u_pred,
        u_ref=u_ref,
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0 / 2.0,
        channel_handling=channel_handling,
        channel_handling_norm=channel_handling_norm,
    )


def mean_MSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    mse_per_sample = jax.vmap(
        lambda p, r: MSE(
            u_pred=p,
            u_ref=r,
            domain_extent=domain_extent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        ),
    )(u_pred, u_ref)
    return jnp.mean(mse_per_sample, axis=0)


def mean_MAE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 1,
) -> Union[float, Float[Array, "C"]]:
    mae_per_sample = jax.vmap(
        lambda p, r: MAE(
            u_pred=p,
            u_ref=r,
            domain_extent=domain_extent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        ),
    )(u_pred, u_ref)
    return jnp.mean(mae_per_sample, axis=0)


def mean_RMSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal["spatial", "norm_before", "norm_after", None] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    rmse_per_sample = jax.vmap(
        lambda p, r: RMSE(
            u_pred=p,
            u_ref=r,
            domain_extent=domain_extent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        ),
    )(u_pred, u_ref)
    return jnp.mean(rmse_per_sample, axis=0)


def mean_nMSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal[
        "spatial", "norm_before", "norm_after", "norm_separate", None
    ] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    nmse_per_sample = jax.vmap(
        lambda p, r: nMSE(
            u_pred=p,
            u_ref=r,
            domain_extent=domain_extent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        ),
    )(u_pred, u_ref)
    return jnp.mean(nmse_per_sample, axis=0)


def mean_nMAE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal[
        "spatial", "norm_before", "norm_after", "norm_separate", None
    ] = "spatial",
    channel_handling_norm: int = 1,
) -> Union[float, Float[Array, "C"]]:
    nmae_per_sample = jax.vmap(
        lambda p, r: nMAE(
            u_pred=p,
            u_ref=r,
            domain_extent=domain_extent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        ),
    )(u_pred, u_ref)
    return jnp.mean(nmae_per_sample, axis=0)


def mean_nRMSE(
    u_pred: Float[Array, "B C ... N"],
    u_ref: Float[Array, "B C ... N"],
    *,
    domain_extent: Optional[float] = None,
    channel_handling: Literal[
        "spatial", "norm_before", "norm_after", "norm_separate", None
    ] = "spatial",
    channel_handling_norm: int = 2,
) -> Union[float, Float[Array, "C"]]:
    nrmse_per_sample = jax.vmap(
        lambda p, r: nRMSE(
            u_pred=p,
            u_ref=r,
            domain_extent=domain_extent,
            channel_handling=channel_handling,
            channel_handling_norm=channel_handling_norm,
        ),
    )(u_pred, u_ref)
    return jnp.mean(nrmse_per_sample, axis=0)
