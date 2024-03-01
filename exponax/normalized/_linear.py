import jax.numpy as jnp
from jaxtyping import Array

from .._base_stepper import BaseStepper
from ..nonlin_fun import ZeroNonlinearFun


class NormalizedLinearStepper(BaseStepper):
    normalized_coefficients: list[float]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients: list[float] = [0.0, -0.5, 0.01],
        dt: float = 1.0,
    ):
        """
        By default: advection-diffusion with normalized advection of 0.5, and
        normalized diffusion of 0.01.

        Take care of the signs!

        Normalized coefficients are alpha_i / L^i, where L is the domain extent.
        """
        self.normalized_coefficients = normalized_coefficients
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(self, derivative_operator: Array) -> Array:
        linear_operator = sum(
            jnp.sum(
                c * (derivative_operator) ** i,
                axis=0,
                keepdims=True,
            )
            for i, c in enumerate(self.normalized_coefficients)
        )
        return linear_operator

    def _build_nonlinear_fun(self, derivative_operator: Array):
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
        )
