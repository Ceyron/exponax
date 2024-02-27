import jax.numpy as jnp

from jax import Array

from ..base_stepper import BaseStepper
from ..nonlinear_functions import GradientNormNonlinearFun
from jaxtyping import Complex, Float, Array


class NormalizedGradientNormStepper(BaseStepper):
    normalized_coefficients: list[float]
    normalized_gradient_norm_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dt: float = 0.1,
        normalized_coefficients: list[float] = [
            0.0, 0.0, -1.0 / (60.0**2), 0.0, -1.0 / (60.0**4)
        ],
        normalized_gradient_norm_scale: float = 1.0 / (60.0**2),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        the number of channels do **not** grow with the number of spatial
        dimensions. They are always 1.

        By default: the KS equation on L=60.0

        **Arguments:**
        - `num_spatial_dims`: number of spatial dimensions
        - `num_points`: number of points in each spatial dimension
        - `dt`: time step (default: 0.1)
        - `normalized_coefficients`: coefficients for the linear operator,
          `normalized_coefficients[i]` is the coefficient for the `i`-th
          derivative 
        - `normalized_gradient_norm_scale`: scale for the gradient norm
        - `order`: order of the derivative operator (default: 2)
        - `dealiasing_fraction`: fraction of the wavenumbers being kept before
          applying any nonlinearity (default: 2/3)
        - `n_circle_points`: number of points to use for the complex contour
          integral when computing coefficients for the exponential time
            differencing Runge Kutta method (default: 16)
        - `circle_radius`: radius of the complex contour integral when computing
            coefficients for the exponential time differencing Runge Kutta method
            (default: 1.0)
        """
        self.normalized_coefficients = normalized_coefficients
        self.normalized_gradient_norm_scale = normalized_gradient_norm_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
            n_circle_points=n_circle_points,
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
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.normalized_gradient_norm_scale,
            zero_mode_fix=True,
        )
