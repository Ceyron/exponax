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
