import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator, space_indices, spatial_shape
from ..nonlin_fun import BaseNonlinearFun


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


class GrayScott(BaseStepper):
    epsilon_1: float
    epsilon_2: float
    b: float
    d: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        epsilon_1: float = 0.00002,
        epsilon_2: float = 0.00001,
        b: float = 0.04,
        d: float = 0.1,
        order: int = 2,
        dealiasing_fraction: float = 1
        / 2,  # Needs lower value due to cubic nonlinearity
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.b = b
        self.d = d
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=2,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "2 ... (N//2)+1"]:
        laplace = build_laplace_operator(derivative_operator, order=2)
        linear_operator = jnp.concatenate(
            [
                self.epsilon_1 * laplace,
                self.epsilon_2 * laplace,
            ]
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> GrayScottNonlinearFun:
        return GrayScottNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            b=self.b,
            d=self.d,
            dealiasing_fraction=self.dealiasing_fraction,
        )
