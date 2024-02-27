import jax.numpy as jnp

from jax import Array

from ..base_stepper import BaseStepper
from ..nonlinear_functions import ConvectionNonlinearFun
from jaxtyping import Complex, Float, Array
from ..spectral import build_laplace_operator, build_gradient_inner_product_operator


class Burgers(BaseStepper):
    diffusivity: float
    convection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.1,
        convection_scale: float = 1.0,
        order=2,
        dealiasing_fraction: float = 2 / 3,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Convection is always scaled by 0.5, use `convection_scale` to multiply
        an additional factor.
        """
        self.diffusivity = diffusivity
        self.convection_scale = convection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=num_spatial_dims,  # Number of channels grows with dimension
            order=order,
            n_circle_points=n_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:

        # The linear operator is the same for all D channels
        return self.diffusivity * build_laplace_operator(derivative_operator)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ConvectionNonlinearFun:
        return ConvectionNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.convection_scale,
        )
