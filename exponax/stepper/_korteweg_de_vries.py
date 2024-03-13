from typing import TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_gradient_inner_product_operator, build_laplace_operator
from ..nonlin_fun import ConvectionNonlinearFun

D = TypeVar("D")


class KortewegDeVries(BaseStepper):
    convection_scale: float
    dispersivity: float
    diffusivity: float
    dealiasing_fraction: float
    advect_over_diffuse: bool
    single_channel: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        convection_scale: float = -6.0,
        dispersivity: float = 1.0,
        advect_over_diffuse: bool = False,
        single_channel: bool = False,
        diffusivity: float = 0.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.convection_scale = convection_scale
        self.dispersivity = dispersivity
        self.advect_over_diffuse = advect_over_diffuse
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
        dispersion_velocity = self.dispersivity * jnp.ones(self.num_spatial_dims)
        laplace_operator = build_laplace_operator(derivative_operator, order=2)
        if self.advect_over_diffuse:
            linear_operator = (
                -build_gradient_inner_product_operator(
                    derivative_operator, self.advect_over_diffuse_dispersivity, order=1
                )
                * laplace_operator
                + self.diffusivity * laplace_operator
            )
        else:
            linear_operator = (
                -build_gradient_inner_product_operator(
                    derivative_operator, dispersion_velocity, order=3
                )
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
