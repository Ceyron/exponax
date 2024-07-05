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

        - The stepper is unconditionally stable, not matter the choice of
            any argument because the equation is solved analytically in Fourier
            space. **However**, note that initial conditions with modes higher
            than the Nyquist freuency (`(N//2)+1` with `N` being the
            `num_points`) lead to spurious oscillations.
        - Ultimately, only the factor `c Δt / L` affects the characteristic
            of the dynamics. See also
            [`exponax.normalized.NormalizedLinearStepper`][] with
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
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `diffusivity` (keyword-only): The diffusivity `ν`. In higher
            dimensions, this can be a scalar (=float), a vector of length `d`,
            or a matrix of shape `d ˣ d`. If a scalar is given, the diffusivity
            is assumed to be the same in all spatial dimensions. If a vector (of
            length `d`) is given, the diffusivity varies across dimensions (=>
            diagonal diffusion). For a matrix, there is fully anisotropic
            diffusion. In this case, `A` must be symmetric positive definite
            (SPD). Default: `0.01`.

        **Notes:**

        - The stepper is unconditionally stable, not matter the choice of
            any argument because the equation is solved analytically in Fourier
            space.
        - A `ν > 0` leads to stable and decaying solutions (i.e., energy is
            removed from the system). A `ν < 0` leads to unstable and growing
            solutions (i.e., energy is added to the system).
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
            self.num_spatial_dims,
            self.num_points,
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
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `velocity` (keyword-only): The advection speed `c`. In higher
            dimensions, this can be a scalar (=float) or a vector of length `d`.
            If a scalar is given, the advection speed is assumed to be the same
            in all spatial dimensions. Default: `1.0`.
        - `diffusivity` (keyword-only): The diffusivity `ν`. In higher
            dimensions, this can be a scalar (=float), a vector of length `d`,
            or a matrix of shape `d ˣ d`. If a scalar is given, the diffusivity
            is assumed to be the same in all spatial dimensions. If a vector (of
            length `d`) is given, the diffusivity varies across dimensions (=>
            diagonal diffusion). For a matrix, there is fully anisotropic
            diffusion. In this case, `A` must be symmetric positive definite
            (SPD). Default: `0.01`.

        **Notes:**

        - The stepper is unconditionally stable, not matter the choice of
            any argument because the equation is solved analytically in Fourier
            space. **However**, note that initial conditions with modes higher
            than the Nyquist freuency (`(N//2)+1` with `N` being the
            `num_points`) lead to spurious oscillations.
        - Ultimately, only the factors `c Δt / L` and `ν Δt / L²` affect the
            characteristic of the dynamics. See also
            [`exponax.normalized.NormalizedLinearStepper`][] with
            `normalized_coefficients = [0, alpha_1, alpha_2]` with `alpha_1 = -
            velocity * dt / domain_extent` and `alpha_2 = diffusivity * dt /
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

        advection_operator = -build_gradient_inner_product_operator(
            derivative_operator, self.velocity, order=1
        )

        linear_operator = advection_operator + diffusion_operator

        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
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

        - The stepper is unconditionally stable, not matter the choice of
            any argument because the equation is solved analytically in Fourier
            space. **However**, note that initial conditions with modes higher
            than the Nyquist freuency (`(N//2)+1` with `N` being the
            `num_points`) lead to spurious oscillations.
        - Ultimately, only the factor `𝒸 Δt / L³` affects the
            characteristic of the dynamics. See also
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
            self.num_spatial_dims,
            self.num_points,
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
        hyper_diffusivity: float = 0.0001,
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
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `hyper_diffusivity` (keyword-only): The hyper-diffusivity `ν`.
            This stepper only supports scalar (=isotropic) hyper-diffusivity.
            Default: 0.0001.
        - `diffuse_on_diffuse` (keyword-only): If `True`, the second form
            of the hyper-diffusion equation in higher dimensions is used. As a
            consequence, there will be mixing in the spatial derivatives.
            Default: `False`.

        **Notes:**

        - The stepper is unconditionally stable, not matter the choice of
            any argument because the equation is solved analytically in Fourier
            space.
        - Ultimately, only the factor `μ Δt / L⁴` affects the characteristic
            of the dynamics. See also
            [`exponax.normalized.NormalizedLinearStepper`][] with
            `normalized_coefficients = [0, 0, 0, 0, alpha_4]` with `alpha_4 = -
            hyper_diffusivity * dt / domain_extent**4`.
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
            self.num_spatial_dims,
            self.num_points,
        )


class GeneralLinearStepper(BaseStepper):
    coefficients: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        coefficients: tuple[float, ...] = (0.0, -0.1, 0.01),
    ):
        """
        General timestepper for a d-dimensional (`d ∈ {1, 2, 3}`) linear
        equation with an arbitrary combination of derivative terms and
        respective coefficients. To simplify the interface, only isotropicity is
        supported. For example, the advection speed is the same in all spatial
        dimensions, or diffusion is equally strong in all spatial dimensions.

        In 1d the equation is given by

        ```
            uₜ = sum_j a_j uₓˢ
        ```

        with `uₓˢ` denoting the s-th derivative of `u` with respect to `x`. The
        coefficient corresponding to this derivative is `a_j`.

        The isotropic version in higher dimensions can expressed as

        ```
            uₜ = sum_j a_j (1⋅∇ʲ)u
        ```

        with `1⋅∇ʲ` denoting the j-th repeated elementwise product of the nabla
        operator with itself in an inner product with the one vector. For
        example, `1⋅∇¹` is the collection of first derivatives, `1⋅∇²` is the
        collection of second derivatives (=Laplace operator), etc.

        The interface to this general stepper is the list of coefficients
        containing the `a_j`. Its length determines the highes occuring order of
        derivative. Note that this list starts at zero. If only one specific
        linear term is wanted, have all prior coefficients set to zero.

        The default configuration is an advection-diffusion equation with `a_0 =
        0`, `a_1 = -0.1`, and `a_2 = 0.01`.

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
            - `coefficients` (keyword-only): The list of coefficients `a_j`
                corresponding to the derivatives. Default: `[0.0, -0.1, 0.01]`.

        **Notes:**
            - There is a repeating pattern in the effect of orders of
              derivatives:
                - Even derivatives (i.e., 0, 2, 4, 6, ...) scale the
                    solution. Order 0 scales all wavenumbers/modes equally (if
                    its coefficient is negative, this is also called a drag).
                    Order 2 scales higher wavenumbers/modes stronger with the
                    dependence on the effect on the wavenumber being
                    quadratically. Order 4 also scales but stronger than order
                    4. Its dependency on the wavenumber is quartically. This
                    pattern continues for higher orders.
                - Odd derivatives (i.e, 1, 3, 5, 7, ...) rotate the solution in
                    Fourier space. In state space, this is observed as
                    advection. Order 1 rotates all wavenumbers/modes equally. In
                    state space, this is observed in that the initial condition
                    just moves over the domain. Order 3 rotates higher
                    wavenumbers/modes stronger with the dependence on the
                    wavenumber being quadratic. If certain wavenumbers are
                    rotated at a different speed, there is still advection in
                    state space but certain patterns in the initial condition
                    are advected at different speeds. As such, it is observed
                    that the shape of the initial condition dissolves. The
                    effect continues for higher orders with the dependency on
                    the wavenumber becoming continuously stronger.
            - Take care of the signs of coefficients. In contrast to the
              indivial linear steppers ([`exponax.stepper.Advection`][],
              [`exponax.stepper.Diffusion`][], etc.), the signs are not
              automatically taken care of to produce meaningful coefficients.
              For the general linear stepper all linear derivatives are on the
              right-hand side of the equation. This has the following effect
              based on the order of derivative (this a consequence of squaring
              the imaginary unit returning -1):
                - Zeroth-Order: A negative coeffcient is a drag and removes
                    energy from the system. A positive coefficient adds energy
                    to the system.
                - First-Order: A negative coefficient rotates the solution
                    clockwise. A positive coefficient rotates the solution
                    counter-clockwise. Hence, negative coefficients advect
                    solutions to the right, positive coefficients advect
                    solutions to the left.
                - Second-Order: A positive coefficient diffuses the solution
                    (i.e., removes energy from the system). A negative
                    coefficient adds energy to the system.
                - Third-Order: A negative coefficient rotates the solution
                    counter-clockwise. A positive coefficient rotates the
                    solution clockwise. Hence, negative coefficients advect
                    solutions to the left, positive coefficients advect
                    solutions to the right.
                - Fourth-Order: A negative coefficient diffuses the solution
                    (i.e., removes energy from the system). A positive
                    coefficient adds energy to the system.
                - ...
            - The stepper is unconditionally stable, no matter the choice of
                any argument because the equation is solved analytically in
                Fourier space. **However**, note if you have odd-order
                derivative terms (e.g., advection or dispersion) and your
                initial condition is **not** bandlimited (i.e., it contains
                modes beyond the Nyquist frequency of `(N//2)+1`) there is a
                chance spurious oscillations can occur.
            - Ultimately, only the factors `a_j Δt / Lʲ` affect the
                characteristic of the dynamics. See also
                [`exponax.normalized.NormalizedLinearStepper`][] with
                `normalized_coefficients = [0, alpha_1, alpha_2, ...]` with
                `alpha_j = coefficients[j] * dt / domain_extent**j`. You can use
                the function [`exponax.normalized.normalize_coefficients`][] to
                obtain the normalized coefficients.
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
            self.num_spatial_dims,
            self.num_points,
        )
