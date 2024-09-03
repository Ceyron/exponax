"""
This is experimental. So far I have been unable to get it to work.
"""
import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ..._spectral import build_laplace_operator
from ...nonlin_fun import BaseNonlinearFun


class BelousovZhabotinskyNonlinearFun(BaseNonlinearFun):
    """
    Taken from: https://github.com/chebfun/chebfun/blob/db207bc9f48278ca4def15bf90591bfa44d0801d/spin.m#L73
    """

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        num_channels = u_hat.shape[0]
        if num_channels != 3:
            raise ValueError("num_channels must be 3")
        u = self.ifft(self.dealias(u_hat))
        u_power = jnp.stack(
            [
                u[0] + u[1] - u[0] * u[1] - u[0] ** 2,
                u[2] - u[1] - u[0] * u[1],
                u[0] - u[2],
            ]
        )
        u_power_hat = self.fft(u_power)
        return u_power_hat


class BelousovZhabotinsky(BaseStepper):
    diffusivities: tuple[float, float, float]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivities: tuple[float, float, float] = (1e-5, 2e-5, 1e-5),
        order: int = 2,
        dealiasing_fraction: float = 1
        / 2,  # Needs lower value due to cubic nonlinearity
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.diffusivities = diffusivities
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=3,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "3 ... (N//2)+1"]:
        laplace = build_laplace_operator(derivative_operator, order=2)
        linear_operator = jnp.concatenate(
            [
                self.diffusivities[0] * laplace,
                self.diffusivities[1] * laplace,
                self.diffusivities[2] * laplace,
            ]
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> BelousovZhabotinskyNonlinearFun:
        return BelousovZhabotinskyNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            dealiasing_fraction=self.dealiasing_fraction,
        )
