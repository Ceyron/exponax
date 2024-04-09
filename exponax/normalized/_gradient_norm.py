import jax.numpy as jnp
from jaxtyping import Array

from .._base_stepper import BaseStepper
from ..nonlin_fun import GradientNormNonlinearFun
from ._utils import (
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_gradient_norm_scale_from_difficulty,
)


class NormalizedGradientNormStepper(BaseStepper):
    normalized_coefficients: tuple[float, ...]
    normalized_gradient_norm_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients: tuple[float, ...] = (
            0.0,
            0.0,
            -1.0 * 0.1 / (60.0**2),
            0.0,
            -1.0 * 0.1 / (60.0**4),
        ),
        normalized_gradient_norm_scale: float = 1.0 * 0.1 / (60.0**2),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        the number of channels do **not** grow with the number of spatial
        dimensions. They are always 1.

        By default: the KS equation on L=60.0

        """
        self.normalized_coefficients = normalized_coefficients
        self.normalized_gradient_norm_scale = normalized_gradient_norm_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
            num_channels=1,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
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
        return GradientNormNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.normalized_gradient_norm_scale,
            zero_mode_fix=True,
        )


class DifficultyGradientNormStepper(NormalizedGradientNormStepper):
    linear_difficulties: tuple[float, ...]
    gradient_norm_difficulty: float

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (0.0, 0.0, -0.128, 0.0, -0.32768),
        gradient_norm_difficulty: float = 0.064,
        maximum_absolute: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default: KS equation
        """
        self.linear_difficulties = linear_difficulties
        self.gradient_norm_difficulty = gradient_norm_difficulty

        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            linear_difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
        normalized_gradient_norm_scale = (
            extract_normalized_gradient_norm_scale_from_difficulty(
                gradient_norm_difficulty,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
                maximum_absolute=maximum_absolute,
            )
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_coefficients=normalized_coefficients,
            normalized_gradient_norm_scale=normalized_gradient_norm_scale,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )
