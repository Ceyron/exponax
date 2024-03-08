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
                - velocity * dt / domain_extent`.
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

        **Notes:**
            - The stepper is unconditionally stable, not matter the choice of
                any argument because the equation is solved analytically in
                Fourier space.
            - A `ν > 0` leads to stable and decaying solutions (i.e., energy is
                removed from the system). A `ν < 0` leads to unstable and
                growing solutions (i.e., energy is added to the system).
            - Ultimately, only the factor `ν Δt / L²` affects the characteristic
                of the dynamics. See also
                [`exponax.normalized.NormalizedLinearStepper`][] with
                `normalized_coefficients = [0, 0, alpha_2]` with `alpha_2 =
                diffusivity * dt / domain_extent**2`.
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
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) advection-diffusion
        equation on periodic boundary conditions.

        In 1d, the advection-diffusion equation is given by

        ```
            uₜ + c uₓ = ν uₓₓ
        ```

        with `c ∈ ℝ` being the velocity/advection speed and `ν ∈ ℝ` being the
        diffusivity.

        In higher dimensions, the advection-diffusion equation can be written as

        ```
            uₜ + c ⋅ ∇u = ν Δu
        ```

        with `c ∈ ℝᵈ` being the velocity/advection vector.

        See also [`exponax.stepper.Diffusion`][] for additional details on
        anisotropic diffusion.

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
            - `diffusivity` (keyword-only): The diffusivity `ν`. In higher
                dimensions, this can be a scalar (=float), a vector of length
                `d`, or a matrix of shape `d ˣ d`. If a scalar is given, the
                diffusivity is assumed to be the same in all spatial dimensions.
                If a vector (of length `d`) is given, the diffusivity varies
                across dimensions (=> diagonal diffusion). For a matrix, there
                is fully anisotropic diffusion. In this case, `A` must be
                symmetric positive definite (SPD). Default: `0.01`.

        **Notes:**
            - The stepper is unconditionally stable, not matter the choice of
                any argument because the equation is solved analytically in
                Fourier space. **However**, note that initial conditions with
                modes higher than the Nyquist freuency (`(N//2)+1` with `N`
                being the `num_points`) lead to spurious oscillations.
            - Ultimately, only the factors `c Δt / L` and `ν Δt / L²` affect the
                characteristic of the dynamics. See also
                [`exponax.normalized.NormalizedLinearStepper`][] with
                `normalized_coefficients = [0, alpha_1, alpha_2]` with `alpha_1 =
                - velocity * dt / domain_extent` and `alpha_2 = diffusivity * dt /
                domain_extent**2`.
        """
        # TODO: more sophisticated checks here
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
                domain. This **includes** the left boundary point and
                **excludes** the right boundary point. In higher dimensions; the
                number of points in each dimension is the same. Hence, the total
                number of degrees of freedom is `Nᵈ`.
            - `dt`: The timestep size `Δt` between two consecutive states.
            - `dispersivity` (keyword-only): The dispersivity `𝒸`. In higher
                dimensions, this can be a scalar (=float) or a vector of length
                `d`. If a scalar is given, the dispersivity is assumed to be the
                same in all spatial dimensions. Default: `1.0`.
            - `advect_on_diffusion` (keyword-only): If `True`, the second form
                of the dispersion equation in higher dimensions is used. As a
                consequence, there will be mixing in the spatial derivatives.
                Default: `False`.

        **Notes:**
            - The stepper is unconditionally stable, not matter the choice of
                any argument because the equation is solved analytically in
                Fourier space. **However**, note that initial conditions with
                modes higher than the Nyquist freuency (`(N//2)+1` with `N`
                being the `num_points`) lead to spurious oscillations.
            - Ultimately, only the factor `𝒸 Δt / L³` affects the characteristic
                of the dynamics. See also
                [`exponax.normalized.NormalizedLinearStepper`][] with
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
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) hyper-diffusion
        equation on periodic boundary conditions. A hyper-diffusion equation
        acts like a diffusion equation but higher wavenumbers/modes are damped
        even faster.

        In 1d, the hyper-diffusion equation is given by

        ```
            uₜ = - μ uₓₓₓₓ
        ```

        with `μ ∈ ℝ` being the hyper-diffusivity.

        Note the minus sign because by default, a fourth-order derivative
        dampens with a negative coefficient. To match the concept of
        second-order diffusion, a negation is introduced.

        In higher dimensions, the hyper-diffusion equation can be written as

        ```
            uₜ = − μ ((∇⊙∇) ⋅ (∇⊙∇)) u
        ```

        or

        ```
            uₜ = - μ Δ(Δu)
        ```

        The latter introduces spatial mixing.

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
            - `hyper_diffusivity` (keyword-only): The hyper-diffusivity `ν`.
                This stepper only supports scalar (=isotropic)
                hyper-diffusivity. Default: 1.0.
            - `diffuse_on_diffuse` (keyword-only): If `True`, the second form
                of the hyper-diffusion equation in higher dimensions is used. As
                a consequence, there will be mixing in the spatial derivatives.
                Default: `False`.

        **Notes:**
            - The stepper is unconditionally stable, not matter the choice of
                any argument because the equation is solved analytically in
                Fourier space.
            - Ultimately, only the factor `μ Δt / L⁴` affects the characteristic
                of the dynamics. See also
                [`exponax.normalized.NormalizedLinearStepper`][] with
                `normalized_coefficients = [0, 0, 0, 0, alpha_4]` with `alpha_4
                = - hyper_diffusivity * dt / domain_extent**4`.
        """
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
