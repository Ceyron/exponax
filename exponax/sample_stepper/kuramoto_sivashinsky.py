import jax.numpy as jnp

from jax import Array

from ..base_stepper import BaseStepper
from ..nonlinear_functions import (
    GradientNormNonlinearFun,
    ConvectionNonlinearFun,
)
from jaxtyping import Complex, Float, Array
from ..spectral import build_laplace_operator, build_gradient_inner_product_operator


class KuramotoSivashinsky(BaseStepper):
    gradient_norm_scale: float
    second_order_diffusivity: float
    fourth_order_diffusivity: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        gradient_norm_scale: float = 1.0,
        second_order_diffusivity: float = 1.0,
        fourth_order_diffusivity: float = 1.0,
        dealiasing_fraction: float = 2 / 3,
        order: int = 2,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Implements the KS equations as used in the combustion community, i.e.,
        with a gradient-norm nonlinearity instead of the convection nonliearity.
        The advantage is that the number of channels is always 1 no matter the
        number of spatial dimensions.
        """
        self.gradient_norm_scale = gradient_norm_scale
        self.second_order_diffusivity = second_order_diffusivity
        self.fourth_order_diffusivity = fourth_order_diffusivity
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
            n_circle_points=n_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        linear_operator = -self.second_order_diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) - self.fourth_order_diffusivity * build_laplace_operator(
            derivative_operator, order=4
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> GradientNormNonlinearFun:
        return GradientNormNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            zero_mode_fix=True,
            scale=self.gradient_norm_scale,
        )


class KuramotoSivashinskyConservative(BaseStepper):
    convection_scale: float
    second_order_diffusivity: float
    fourth_order_diffusivity: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        convection_scale: float = 1.0,
        second_order_diffusivity: float = 1.0,
        fourth_order_diffusivity: float = 1.0,
        dealiasing_fraction: float = 2 / 3,
        order: int = 2,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Using the fluid dynamics form of the KS equation (i.e. similar to the
        Burgers equation). This also means that the number of channels grow with
        the number of spatial dimensions.
        """
        self.convection_scale = convection_scale
        self.second_order_diffusivity = second_order_diffusivity
        self.fourth_order_diffusivity = fourth_order_diffusivity
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=num_spatial_dims,
            order=order,
            n_circle_points=n_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        linear_operator = -self.second_order_diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) - self.fourth_order_diffusivity * build_laplace_operator(
            derivative_operator, order=4
        )
        return linear_operator

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
            zero_mode_fix=True,
            scale=self.convection_scale,
        )
