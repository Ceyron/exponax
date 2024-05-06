import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import PolynomialNonlinearFun
from ._utils import extract_normalized_coefficients_from_difficulty


class NormalizedPolynomialStepper(BaseStepper):
    normalized_coefficients: tuple[float, ...]
    normalized_polynomial_scales: tuple[float, ...]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients: tuple[float, ...] = (
            10.0 * 0.001 / (10.0**0),
            0.0,
            1.0 * 0.001 / (10.0**2),
        ),
        normalized_polynomial_scales: tuple[float, ...] = (
            0.0,
            0.0,
            -10.0 * 0.001,
        ),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default: Fisher-KPP
        """
        self.normalized_coefficients = normalized_coefficients
        self.normalized_polynomial_scales = normalized_polynomial_scales
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

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        linear_operator = sum(
            jnp.sum(
                c * (derivative_operator) ** i,
                axis=0,
                keepdims=True,
            )
            for i, c in enumerate(self.normalized_coefficients)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> PolynomialNonlinearFun:
        return PolynomialNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            coefficients=self.normalized_polynomial_scales,
            dealiasing_fraction=self.dealiasing_fraction,
        )


class DifficultyPolynomialStepper(NormalizedPolynomialStepper):
    linear_difficulties: tuple[float, ...]
    polynomial_difficulties: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (
            10.0 * 0.001 / (10.0**0) * 48**0,
            0.0,
            1.0 * 0.001 / (10.0**2) * 48**2 * 2**1,
        ),
        polynomial_difficulties: tuple[float, ...] = (
            0.0,
            0.0,
            -10.0 * 0.001,
        ),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default: Fisher-KPP
        """
        self.linear_difficulties = linear_difficulties
        self.polynomial_difficulties = polynomial_difficulties

        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            linear_difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
        # For polynomial nonlinearities, we have difficulties == normalized scales
        normalized_polynomial_scales = polynomial_difficulties

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_coefficients=normalized_coefficients,
            normalized_polynomial_scales=normalized_polynomial_scales,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )
