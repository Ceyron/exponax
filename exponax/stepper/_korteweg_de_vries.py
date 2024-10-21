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
    hyper_diffusivity: float
    dealiasing_fraction: float
    advect_over_diffuse: bool
    diffuse_over_diffuse: bool
    single_channel: bool
    conservative: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        convection_scale: float = -6.0,
        diffusivity: float = 0.0,
        dispersivity: float = 1.0,
        hyper_diffusivity: float = 0.01,
        advect_over_diffuse: bool = False,
        diffuse_over_diffuse: bool = False,
        single_channel: bool = False,
        conservative: bool = False,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) (hyper-viscous)
        Korteweg-de Vries equation on periodic boundary conditions.

        In 1d, the Korteweg-de Vries equation is given by

        ```
            uₜ + b₁ 1/2 (u²)ₓ + a₃ uₓₓₓ = ν uₓₓ - μ uₓₓₓₓ
        ```

        with `b₁` the convection coefficient, `a₃` the dispersion coefficient
        and `ν` the diffusivity. Oftentimes `b₁ = -6` and `ν = 0`. The
        nonlinearity is similar to the Burgers equation and the number of
        channels grow with the number of spatial dimensions. In higher
        dimensions, the equation reads (using vector format for the channels)

        ```
            uₜ + b₁ 1/2 ∇ ⋅ (u ⊗ u) + a₃ 1 ⋅ (∇⊙∇⊙(∇u)) = ν Δu - μ ((∇⊙∇) ⋅ (∇⊙∇)) u
        ```

        or

        ```
            uₜ + b₁ 1/2 ∇ ⋅ (u ⊗ u) + a₃ ∇ ⋅ ∇(Δu) = ν Δu - μ Δ(Δu)
        ```

        if `advect_over_diffuse` is `True` and `diffuse_on_diffuse` is `True`,

        In 1d, the expected temporal behavior is that the initial condition
        breaks into soliton waves that propagate at a speed depending on their
        height. They interact with other soliton waves by being spatially
        displaced but having an unchanged shape and propagation speed. If either
        the `diffusivity` or the `hyper_diffusivity` is non-zero (by default,
        the latter is active), the solution will decay over time with the
        produced small-scall features decaying the fastest.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `convection_scale`: The convection coefficient `b₁`. Note that the
            convection is already scaled by 1/2 to account for the conservative
            evaluation. The value of `b₁` scales it further. Oftentimes `b₁ =
            -6` to match the analytical soliton solutions. See also
            https://en.wikipedia.org/wiki/Korteweg%E2%80%93De_Vries_equation#One-soliton_solution
        - `diffusivity`: The rate at which the solution decays.
        - `dispersivity`: The dispersion coefficient `a₃`. Dispersion refers
            to wavenumber-dependent advection, i.e., higher wavenumbers are
            advected faster. Default `1.0`,
        - `hyper_diffusivity`: The hyper-diffusivity (associated with a
            fourth-order term). This stepper only supports scalar (=isotropic)
            hyper-diffusivity. Default: 0.01.
        - `advect_over_diffuse`: If `True`, the dispersion is computed as
            advection over diffusion. This adds spatial mixing. Default is
            `False`.
        - `diffuse_on_diffuse`: If `True`, the hyper-diffusion is computed as
            the diffusion of the diffusion. This adds spatial mixing. Default is
            `False`.
        - `single_channel`: Whether to use the single channel mode in higher
            dimensions. In this case the the convection is `b₁ (∇ ⋅ 1)(u²)`. In
            this case, the state always has a single channel, no matter the
            spatial dimension. Default: False.
        - `conservative`: Whether to use the conservative form of the convection
            term. Default: False.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for higher
            accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. The default 2/3 corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2. Default:
            2/3.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.

        **Notes:**

            -

        **Good Values:**

        - There is an anlytical solution to the (inviscid, `ν = 0`) KdV of
            `u(t, x) = - 1/2 c^2 sech^2(c/2 (x - ct - a))` with the hyperbolic
            secant `sech` and arbitrarily selected speed `c` and shift `a`.
        - For a nice simulation with an initial condition that breaks into
            solitons choose `domain_extent=20.0` and an initial condition with
            the first 5-10 modes. Set dt=0.01, num points in the range of 50-200
            are sufficient.
        """
        self.convection_scale = convection_scale
        self.diffusivity = diffusivity
        self.dispersivity = dispersivity
        self.hyper_diffusivity = hyper_diffusivity
        self.advect_over_diffuse = advect_over_diffuse
        self.diffuse_over_diffuse = diffuse_over_diffuse
        self.single_channel = single_channel
        self.conservative = conservative
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

        diffusion_operator = self.diffusivity * laplace_operator

        if self.advect_over_diffuse:
            dispersion_operator = (
                -build_gradient_inner_product_operator(
                    derivative_operator, self.advect_over_diffuse_dispersivity, order=1
                )
                * laplace_operator
            )
        else:
            dispersion_operator = -build_gradient_inner_product_operator(
                derivative_operator, dispersion_velocity, order=3
            )

        if self.diffuse_over_diffuse:
            hyper_diffusion_operator = (
                -self.hyper_diffusivity * laplace_operator * laplace_operator
            )
        else:
            hyper_diffusion_operator = -self.hyper_diffusivity * build_laplace_operator(
                derivative_operator, order=4
            )

        linear_operator = (
            diffusion_operator + dispersion_operator + hyper_diffusion_operator
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
            conservative=self.conservative,
        )
