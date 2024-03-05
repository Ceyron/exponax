import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import PolynomialNonlinearFun


class NormalizedPolynomialStepper(BaseStepper):
    normalized_coefficients: list[float]
    normalized_polynomial_scales: list[float]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients: list[float] = [
            10.0 * 0.001 / (10.0**0),
            0.0,
            1.0 * 0.001 / (10.0**2),
        ],
        normalized_polynomial_scales: list[float] = [
            0.0,
            0.0,
            10.0 * 0.001,
        ],
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        n_circle_points: int = 16,
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
            n_circle_points=n_circle_points,
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
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            coefficients=self.normalized_polynomial_scales,
            dealiasing_fraction=self.dealiasing_fraction,
        )
