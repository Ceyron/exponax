"""
Nonlinear terms as they are found in reaction-diffusion(-advection) equations.
"""

import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._spectral import build_laplace_operator, space_indices, spatial_shape
from ._base import BaseNonlinearFun


class GrayScottNonlinearFun(BaseNonlinearFun):
    b: float
    d: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        b: float,
        d: float,
    ):
        if num_channels != 2:
            raise ValueError(f"Expected num_channels = 2, got {num_channels}.")
        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.b = b
        self.d = d

    def evaluate(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_hat_dealiased = self.dealiasing_mask * u_hat
        u = jnp.fft.irfftn(
            u_hat_dealiased,
            s=spatial_shape(self.num_spatial_dims, self.num_points),
            axes=space_indices(self.num_spatial_dims),
        )
        u_power = jnp.stack(
            [
                self.b * (1 - u[0]) - u[0] * u[1] ** 2,
                -self.d * u[1] + u[0] * u[1] ** 2,
            ]
        )
        u_power_hat = jnp.fft.rfftn(u_power, axes=space_indices(self.num_spatial_dims))
        return u_power_hat


class CahnHilliardNonlinearFun(BaseNonlinearFun):
    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        if num_channels != 1:
            raise ValueError(f"Expected num_channels = 1, got {num_channels}.")
        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

    def evaluate(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_hat_dealiased = self.dealiasing_mask * u_hat
        u = jnp.fft.irfftn(
            u_hat_dealiased,
            s=spatial_shape(self.num_spatial_dims, self.num_points),
            axes=space_indices(self.num_spatial_dims),
        )
        u_power = u[0] ** 3
        u_power_hat = jnp.fft.rfftn(u_power, axes=space_indices(self.num_spatial_dims))
        u_power_laplace_hat = (
            build_laplace_operator(self.derivative_operator, order=2) * u_power_hat
        )
        return u_power_laplace_hat


class BelousovZhabotinskyNonlinearFun(BaseNonlinearFun):
    """
    Taken from: https://github.com/chebfun/chebfun/blob/db207bc9f48278ca4def15bf90591bfa44d0801d/spin.m#L73
    """

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        if num_channels != 3:
            raise ValueError(f"Expected num_channels = 3, got {num_channels}.")
        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

    def evaluate(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_hat_dealiased = self.dealiasing_mask * u_hat
        u = jnp.fft.irfftn(
            u_hat_dealiased,
            s=spatial_shape(self.num_spatial_dims, self.num_points),
            axes=space_indices(self.num_spatial_dims),
        )
        u_power = jnp.stack(
            [
                u[0] + u[1] - u[0] * u[1] - u[0] ** 2,
                u[2] - u[1] - u[0] * u[1],
                u[0] - u[2],
            ]
        )
        u_power_hat = jnp.fft.rfftn(u_power, axes=space_indices(self.num_spatial_dims))
        return u_power_hat
