from typing import TypeVar, Union

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from .._spectral import build_gradient_inner_product_operator, build_laplace_operator
from ..nonlin_fun import ConvectionNonlinearFun

D = TypeVar("D")


class KortewegDeVries(BaseStepper):
    convection_scale: float
    pure_dispersivity: Float[Array, "D"]
    advect_over_diffuse_dispersivity: Float[Array, "D"]
    diffusivity: float
    dealiasing_fraction: float
    single_channel: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        convection_scale: float = -6.0,
        pure_dispersivity: Union[Float[Array, "D"], float] = 1.0,
        single_channel: bool = False,
        advect_over_diffuse_dispersivity: Union[Float[Array, "D"], float] = 0.0,
        diffusivity: float = 0.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.convection_scale = convection_scale
        if isinstance(pure_dispersivity, float):
            pure_dispersivity = jnp.ones(num_spatial_dims) * pure_dispersivity
        if isinstance(advect_over_diffuse_dispersivity, float):
            advect_over_diffuse_dispersivity = (
                jnp.ones(num_spatial_dims) * advect_over_diffuse_dispersivity
            )
        self.pure_dispersivity = pure_dispersivity
        self.advect_over_diffuse_dispersivity = advect_over_diffuse_dispersivity
        self.diffusivity = diffusivity
        self.single_channel = single_channel
        self.dealiasing_fraction = dealiasing_fraction

        if single_channel:
            num_channels = 1
        else:
            # number of channels grow with the spatial dimension
            num_channels = num_spatial_dims

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=num_channels,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        laplace_operator = build_laplace_operator(derivative_operator, order=2)
        linear_operator = (
            -build_gradient_inner_product_operator(
                derivative_operator, self.pure_dispersivity, order=3
            )
            - build_gradient_inner_product_operator(
                derivative_operator, self.advect_over_diffuse_dispersivity, order=1
            )
            * laplace_operator
            + self.diffusivity * laplace_operator
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ConvectionNonlinearFun:
        return ConvectionNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.convection_scale,
            single_channel=self.single_channel,
        )
