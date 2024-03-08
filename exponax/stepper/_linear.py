from typing import TypeVar, Union

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from .._spectral import build_gradient_inner_product_operator, build_laplace_operator
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
                domain. This **includes** the left boundary point and
                **excludes** the right boundary point. In higher dimensions; the
                number of points in each dimension is the same. Hence, the total
                number of degrees of freedom is `Nᵈ`.
            - `dt`: The timestep size `Δt` between two consecutive states.
            - `velocity` (keyword-only): The advection speed `c`. In higher
                dimensions, this can be a scalar (=float) or a vector of length
                `d`. If a scalar is given, the advection speed is assumed to be
                the same in all spatial dimensions. Default: `1.0`.

        **Notes:**
            - The stepper is unconditionally stable, not matter the choice of
                any argument because the equation is solved analytically in
                Fourier space. **However**, note that initial conditions with
                modes higher than the Nyquist freuency (`(N//2)+1` with `N`
                being the `num_points`) lead to spurious oscillations.
            - Ultimately, only the factor `c Δt / L` affects the characteristic
                of the dynamics. See also
                [`exponax.normalized.NormalizedLinearStepper`][] with
                `normalized_coefficients = [0, alpha_1]` with `alpha_1 =
                velocity * dt / domain_extent`.q
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
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=1.0,
        )


class Diffusion(BaseStepper):
    diffusivity: Float[Array, "D D"]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: Union[
            Float[Array, "D D"],
            Float[Array, "D"],
            float,
        ] = 0.01,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) diffusion equation
        on periodic boundary conditions.

        In 1d, the diffusion equation is given by

        ```
            uₜ = ν uₓₓ
        ```
        
        with `ν ∈ ℝ` being the diffusivity.

        In higher dimensions, the diffusion equation can written using the
        Laplacian operator.

        ```
            uₜ = ν Δu
        ```

        More generally speaking, there can be anistropic diffusivity given by a
        `A ∈ ℝᵈ ˣ ᵈ` sandwiched between the gradient and divergence operators.

        ```
            uₜ = ∇ ⋅ (A ∇u)
        ```

        **Arguments:**
            - `num_spatial_dims`: The number of spatial dimensions `d`.
            - `domain_extent`: The size of the domain `L`; in higher dimensions
                the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
            - `num_points`: The number of points `N` used to discretize the
                domain. This **includes** the left boundary point and
                **excludes** the right boundary point. In higher dimensions; the
                number of points in each dimension is the same. Hence, the total
                number of degrees of freedom is `Nᵈ`.
            - `dt`: The timestep size `Δt` between two consecutive states.
            - `diffusivity` (keyword-only): The diffusivity `ν`. In higher
                dimensions, this can be a scalar (=float), a vector of length
                `d`, or a matrix of shape `d ˣ d`. If a scalar is given, the
                diffusivity is assumed to be the same in all spatial dimensions.
                If a vector (of length `d`) is given, the diffusivity varies
                across dimensions (=> diagonal diffusion). For a matrix, there
                is fully anisotropic diffusion. In this case, `A` must be
                symmetric positive definite (SPD). Default: `0.01`.
        """
        # ToDo: more sophisticated checks here
        if isinstance(diffusivity, float):
            diffusivity = jnp.diag(jnp.ones(num_spatial_dims)) * diffusivity
        elif len(diffusivity.shape) == 1:
            diffusivity = jnp.diag(diffusivity)
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
        laplace_outer_producct = (
            derivative_operator[:, None] * derivative_operator[None, :]
        )
        linear_operator = jnp.einsum(
            "ij,ij...->...",
            self.diffusivity,
            laplace_outer_producct,
        )
        # Add the necessary singleton channel axis
        linear_operator = linear_operator[None, ...]
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


class AdvectionDiffusion(BaseStepper):
    velocity: Float[Array, "D"]
    diffusivity: Float[Array, "D D"]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        velocity: Union[Float[Array, "D"], float] = 1.0,
        diffusivity: Union[
            Float[Array, "D D"],
            Float[Array, "D"],
            float,
        ] = 0.01,
    ):
        # ToDo: more sophisticated checks here
        if isinstance(velocity, float):
            velocity = jnp.ones(num_spatial_dims) * velocity
        self.velocity = velocity
        if isinstance(diffusivity, float):
            diffusivity = jnp.diag(jnp.ones(num_spatial_dims)) * diffusivity
        elif len(diffusivity.shape) == 1:
            diffusivity = jnp.diag(diffusivity)
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
        laplace_outer_producct = (
            derivative_operator[:, None] * derivative_operator[None, :]
        )
        diffusion_operator = jnp.einsum(
            "ij,ij...->...",
            self.diffusivity,
            laplace_outer_producct,
        )
        # Add the necessary singleton channel axis
        diffusion_operator = diffusion_operator[None, ...]

        advection_operator = - build_gradient_inner_product_operator(
            derivative_operator, self.velocity, order=1
        )

        linear_operator = advection_operator + diffusion_operator

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
