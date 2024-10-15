from typing import TypeVar, Union

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from .._spectral import build_gradient_inner_product_operator, build_laplace_operator
from ..nonlin_fun import ZeroNonlinearFun

D = TypeVar("D")


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
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) dispersion equation
        on periodic boundary conditions. Essentially, a dispersion equation is
        an advection equation with different velocities (=advection speeds) for
        different wavenumbers/modes. Higher wavenumbers/modes are advected
        faster.

        In 1d, the dispersion equation is given by

        ```
            uₜ = 𝒸 uₓₓₓ
        ```

        with `𝒸 ∈ ℝ` being the dispersivity.

        In higher dimensions, the dispersion equation can be written as

        ```
            uₜ = 𝒸 ⋅ (∇⊙∇⊙(∇u))
        ```

        or

        ```
            uₜ = 𝒸 ⋅ ∇(Δu)
        ```

        with `𝒸 ∈ ℝᵈ` being the dispersivity vector

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
        - `dispersivity` (keyword-only): The dispersivity `𝒸`. In higher
            dimensions, this can be a scalar (=float) or a vector of length `d`.
            If a scalar is given, the dispersivity is assumed to be the same in
            all spatial dimensions. Default: `1.0`.
        - `advect_on_diffusion` (keyword-only): If `True`, the second form
            of the dispersion equation in higher dimensions is used. As a
            consequence, there will be mixing in the spatial derivatives.
            Default: `False`.

        **Notes:**

        - The stepper is unconditionally stable, no matter the choice of
            any argument because the equation is solved analytically in Fourier
            space. **However**, note that initial conditions with modes higher
            than the Nyquist freuency (`(N//2)+1` with `N` being the
            `num_points`) lead to spurious oscillations.
        - Ultimately, only the factor `𝒸 Δt / L³` affects the
            characteristic of the dynamics. See also
            [`exponax.stepper.generic.NormalizedLinearStepper`][] with
            `normalized_coefficients = [0, 0, 0, alpha_3]` with `alpha_3 =
            dispersivity * dt / domain_extent**3`.
        """
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
            self.num_spatial_dims,
            self.num_points,
        )
