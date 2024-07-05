import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import GeneralNonlinearFun
from ._utils import (
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_nonlinear_scales_from_difficulty,
)


class NormalizedGeneralNonlinearStepper(BaseStepper):
    normalized_coefficients_linear: tuple[float, ...]
    normalized_coefficients_nonlinear: tuple[float, ...]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients_linear: tuple[float, ...] = (0.0, 0.0, 0.1 * 0.1),
        normalized_coefficients_nonlinear: tuple[float, ...] = (0.0, -1.0 * 0.1, 0.0),
        order=2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default Burgers.
        """

        if len(normalized_coefficients_nonlinear) != 3:
            raise ValueError(
                "The nonlinear coefficients list must have exactly 3 elements"
            )
        self.normalized_coefficients_linear = normalized_coefficients_linear
        self.normalized_coefficients_nonlinear = normalized_coefficients_nonlinear
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
            for i, c in enumerate(self.normalized_coefficients_linear)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> GeneralNonlinearFun:
        return GeneralNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale_list=self.normalized_coefficients_nonlinear,
            zero_mode_fix=True,  # ToDo: check this
        )


class DifficultyGeneralNonlinearStepper(NormalizedGeneralNonlinearStepper):
    linear_difficulties: tuple[float, ...]
    nonlinear_difficulties: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (
            0.0,
            0.0,
            0.1 * 0.1 / 1.0 * 48**2 * 2,
        ),
        nonlinear_difficulties: tuple[float, ...] = (
            0.0,
            -1.0 * 0.1 / 1.0 * 48,
            0.0,
        ),
        maximum_absolute: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default Burgers.
        """
        self.linear_difficulties = linear_difficulties
        self.nonlinear_difficulties = nonlinear_difficulties

        normalized_coefficients_linear = (
            extract_normalized_coefficients_from_difficulty(
                linear_difficulties,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
            )
        )
        normalized_coefficients_nonlinear = (
            extract_normalized_nonlinear_scales_from_difficulty(
                nonlinear_difficulties,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
                maximum_absolute=maximum_absolute,
            )
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_coefficients_linear=normalized_coefficients_linear,
            normalized_coefficients_nonlinear=normalized_coefficients_nonlinear,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )
