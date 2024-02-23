from typing import Union

import jax.numpy as jnp

from jax import Array

from ..base_stepper import BaseStepper
from ..nonlinear_functions import ZeroNonlinearFun
from jaxtyping import Complex, Float, Array
from ..spectral import build_laplace_operator, build_gradient_inner_product_operator


class Advection(BaseStepper):
    velocity: Float[Array, "D"]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        velocity: Union[Float[Array, "D"], float] = 1.0,
    ):
        if isinstance(velocity, float):
            velocity = jnp.ones(num_spatial_dims) * velocity
        self.velocity = velocity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        # Requires minus to move term to the rhs
        return -build_gradient_inner_product_operator(
            derivative_operator, self.velocity, order=1
        )

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=1.0,
        )


class Diffusion(BaseStepper):
    diffusivity: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
    ):
        self.diffusivity = diffusivity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        return self.diffusivity * build_laplace_operator(derivative_operator)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=1.0,
        )


class AdvectionDiffusion(BaseStepper):
    velocity: Float[Array, "D"]
    diffusivity: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        velocity: Union[Float[Array, "D"], float] = 1.0,
        diffusivity: float = 0.01,
    ):
        if isinstance(velocity, float):
            velocity = jnp.ones(num_spatial_dims) * velocity
        self.velocity = velocity
        self.diffusivity = diffusivity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        return -build_gradient_inner_product_operator(
            derivative_operator, self.velocity, order=1
        ) + self.diffusivity * build_laplace_operator(derivative_operator)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=1.0,
        )


class Dispersion(BaseStepper):
    dispersivity: Float[Array, "D"]
    advect_on_diffusion: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        dispersivity: Union[Float[Array, "D"], float] = 1.0,
        advect_on_diffusion: bool = False,
    ):
        if isinstance(dispersivity, float):
            dispersivity = jnp.ones(num_spatial_dims) * dispersivity
        self.dispersivity = dispersivity
        self.advect_on_diffusion = advect_on_diffusion
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        if self.advect_on_diffusion:
            laplace_operator = build_laplace_operator(derivative_operator)
            advection_operator = build_gradient_inner_product_operator(
                derivative_operator, self.dispersivity, order=1
            )
            linear_operator = advection_operator * laplace_operator
        else:
            linear_operator = build_gradient_inner_product_operator(
                derivative_operator, self.dispersivity, order=3
            )

        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=1.0,
        )


class HyperDiffusion(BaseStepper):
    hyper_diffusivity: float
    diffuse_on_diffuse: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        hyper_diffusivity: float = 1.0,
        diffuse_on_diffuse: bool = False,
    ):
        self.hyper_diffusivity = hyper_diffusivity
        self.diffuse_on_diffuse = diffuse_on_diffuse
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        # Use minus sign to have diffusion work in "correct direction" by default
        if self.diffuse_on_diffuse:
            laplace_operator = build_laplace_operator(derivative_operator)
            linear_operator = (
                -self.hyper_diffusivity * laplace_operator * laplace_operator
            )
        else:
            linear_operator = -self.hyper_diffusivity * build_laplace_operator(
                derivative_operator, order=4
            )

        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=1.0,
        )


class GeneralLinearStepper(BaseStepper):
    coefficients: list[float]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        coefficients: list[float] = [0.0, -0.1, 0.01],
    ):
        """
        Isotropic linear operators!

        By default: advection-diffusion with advection of 0.1 and diffusion of
        0.01.

        Take care of the signs!
        """
        self.coefficients = coefficients
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
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
            for i, c in enumerate(self.coefficients)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=1.0,
        )
