import jax.numpy as jnp
from jaxtyping import Array

from .._base_stepper import BaseStepper
from ..nonlin_fun import ZeroNonlinearFun
from ._utils import extract_normalized_coefficients_from_difficulty


class NormalizedLinearStepper(BaseStepper):
    normalized_coefficients: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients: tuple[float, ...] = (0.0, -0.5, 0.01),
    ):
        """
        By default: advection-diffusion with normalized advection of 0.5, and
        normalized diffusion of 0.01.

        Take care of the signs!

        Normalized coefficients are alpha_i * dt / L^i, where dt is the time
        step size and L is the domain extent.
        """
        self.normalized_coefficients = normalized_coefficients
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
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
        )


class DifficultyLinearStepper(NormalizedLinearStepper):
    difficulties: tuple[float, ...]

    def __init__(
        self,
        *,
        difficulties: tuple[float, ...] = (0.0, -2.0),
        num_spatial_dims: int = 1,
        num_points: int = 48,
    ):
        """
        By default: Advection equation with CFL number 2 on 48 points resolution
        in one spatial dimension.
        """
        self.difficulties = difficulties
        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_coefficients=normalized_coefficients,
        )


class DiffultyLinearStepperSimple(DifficultyLinearStepper):
    def __init__(
        self,
        *,
        difficulty: float = -2.0,
        order: int = 1,
        num_spatial_dims: int = 1,
        num_points: int = 48,
    ):
        difficulties = (0.0,) * (order - 1) + (difficulty,)
        super().__init__(
            difficulties=difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
