from typing import TypeVar, Union

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from .._spectral import build_gradient_inner_product_operator
from ..nonlin_fun import ZeroNonlinearFun

D = TypeVar("D")


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
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) advection equation
        on periodic boundary conditions.

        In 1d, the advection equation is given by

        ```
            uₜ + c uₓ = 0
        ```

        with `c ∈ ℝ` being the velocity/advection speed.

        In higher dimensions, the advection equation can written as the inner
        product between velocity vector and gradient

        ```
            uₜ + c ⋅ ∇u = 0
        ```

        with `c ∈ ℝᵈ` being the velocity/advection vector.

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
        - `velocity` (keyword-only): The advection speed `c`. In higher
            dimensions, this can be a scalar (=float) or a vector of length `d`.
            If a scalar is given, the advection speed is assumed to be the same
            in all spatial dimensions. Default: `1.0`.

        **Notes:**

        - The stepper is unconditionally stable, no matter the choice of
            any argument because the equation is solved analytically in Fourier
            space. **However**, note that initial conditions with modes higher
            than the Nyquist freuency (`(N//2)+1` with `N` being the
            `num_points`) lead to spurious oscillations.
        - Ultimately, only the factor `c Δt / L` affects the characteristic
            of the dynamics. See also
            [`exponax.stepper.generic.NormalizedLinearStepper`][] with
            `normalized_coefficients = [0, alpha_1]` with `alpha_1 = - velocity
            * dt / domain_extent`.
        """
        # TODO: better checks on the desired type of velocity
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
            self.num_spatial_dims,
            self.num_points,
        )
