import jax.numpy as jnp

from jax import Array

from ..base_stepper import BaseStepper
from ..nonlinear_functions import ConvectionNonlinearFun
from jaxtyping import Complex, Float, Array


class NormalizedConvectionStepper(BaseStepper):
    normalized_coefficients: list[float]
    normalized_convection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dt: float = 0.1,
        normalized_coefficients: list[float] = [0.0, 0.0, 0.01],
        normalized_convection_scale: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default: Behaves like a Burgers with

        ``` Burgers(
            D=D, L=1, N=N, dt=dt, diffusivity=0.01,
        )
        ```

        If you set `L=2 * jnp.pi` of your unnormalized scenario, then you have
        to set your coefficients to `alpha_i * dt` (make sure to use the same dt
        as is used here as the keyword based argument).

        If you set `L=1` of your unnormalized scenario, then you have to set
        your coefficients to `alpha_i * dt / (2 * jnp.pi)^s` (make sure to use
        the same dt as is used here as the keyword based argument) **and** set
        your convection scale to whatever you had prior multiplied by 2 *
        jnp.pi.

        If you set `L=L` of your unnormalized scenario, then you have to set
        your coefficients to `alpha_i * dt * (L / (2 * jnp.pi))^s` (make sure to
        use the same dt as is used here as the keyword based argument) **and**
        set your convection scale to whatever you had prior multiplied by 2 *
        jnp.pi / L.

        number of channels grow with number of spatial dimensions

        **Arguments:**

        - `num_spatial_dims`: number of spatial dimensions
        - `num_points`: number of points in each spatial dimension
        - `dt`: time step (default: 0.1)
        - `normalized_coefficients`: coefficients for the linear operator,
          `normalized_coefficients[i]` is the coefficient for the `i`-th
          derivative (default: [0.0, 0.0, 0.01 * 0.1] refers to a diffusion
          (2nd) order term)
        - `normalized_convection_scale`: convection scale for the nonlinear
            function (default: 1.0)
        - `order`: order of exponential time differencing Runge Kutta method,
          can be 1, 2, 3, 4 (default: 2)
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
        self.normalized_convection_scale = normalized_convection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=dt,
            num_channels=num_spatial_dims,
            order=order,
            n_circle_points=n_circle_points,
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
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.normalized_convection_scale,
        )
