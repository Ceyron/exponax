import jax.numpy as jnp
from jaxtyping import Array

from .._base_stepper import BaseStepper
from ..nonlin_fun import ConvectionNonlinearFun
from ._utils import (
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_convection_scale_from_difficulty,
)


class NormalizedConvectionStepper(BaseStepper):
    normalized_coefficients: tuple[float, ...]
    normalized_convection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients: tuple[float, ...] = (0.0, 0.0, 0.01 * 0.1),
        normalized_convection_scale: float = 1.0 * 0.1,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default: Behaves like a Burgers with

        ``` Burgers(
            D=D, L=1, N=N, dt=0.1, diffusivity=0.01,
        )
        ```
        """
        self.normalized_coefficients = normalized_coefficients
        self.normalized_convection_scale = normalized_convection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
            num_channels=num_spatial_dims,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(self, derivative_operator: Array) -> Array:
        # Now the linear operator is unscaled
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
        return ConvectionNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.normalized_convection_scale,
        )


class DifficultyConvectionStepper(NormalizedConvectionStepper):
    linear_difficulties: tuple[float, ...]
    convection_difficulty: float

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (0.0, 0.0, 4.5),
        convection_difficulty: float = 5.0,
        maximum_absolute: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default: Behaves like a Burgers

        """
        self.linear_difficulties = linear_difficulties
        self.convection_difficulty = convection_difficulty
        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            linear_difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
        normalized_convection_scale = (
            extract_normalized_convection_scale_from_difficulty(
                convection_difficulty,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
                maximum_absolute=maximum_absolute,
            )
        )
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_coefficients=normalized_coefficients,
            normalized_convection_scale=normalized_convection_scale,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )
