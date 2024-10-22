from typing import TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ...nonlin_fun import ZeroNonlinearFun
from ._utils import extract_normalized_coefficients_from_difficulty

D = TypeVar("D")


class GeneralLinearStepper(BaseStepper):
    linear_coefficients: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        linear_coefficients: tuple[float, ...] = (0.0, -0.1, 0.01),
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
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `linear_coefficients` (keyword-only): The list of coefficients `a_j`
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
            [`exponax.stepper.generic.NormalizedLinearStepper`][] with
            `normalized_coefficients = [0, alpha_1, alpha_2, ...]` with
            `alpha_j = coefficients[j] * dt / domain_extent**j`. You can use
            the function
            [`exponax.stepper.generic.normalize_coefficients`][] to obtain
            the normalized coefficients.
        """
        self.linear_coefficients = linear_coefficients
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
            for i, c in enumerate(self.linear_coefficients)
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


class NormalizedLinearStepper(GeneralLinearStepper):
    normalized_linear_coefficients: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_linear_coefficients: tuple[float, ...] = (0.0, -0.5, 0.01),
    ):
        """
        Timestepper for d-dimensional (`d ∈ {1, 2, 3}`) linear PDEs on periodic
        boundary conditions with normalized dynamics.

        If the PDE in physical description is

            uₜ = ∑ᵢ aᵢ (∂ₓ)ⁱ u

        with `aᵢ` the coefficients on `domain_extent` `L` with time step `dt`,
        the normalized coefficients are

            αᵢ = (aᵢ Δt)/(Lⁱ)

        Important: note that the `domain_extent` is raised to the order of
        linear derivative `i`.

        One can also think of normalized dynamics, as a PDE on `domain_extent`
        `1.0` with time step `dt=1.0`.

        Take care of the signs!

        In the defaulf configuration of this timestepper, the PDE is an
        advection-diffusion equation with normalized advection of 0.5 and
        normalized diffusion of 0.01.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `normalized_coefficients`: The coefficients of the normalized
            dynamics. This must a tuple of floats. The length of the tuple
            defines the highest occuring linear derivative in the PDE.
        """
        self.normalized_linear_coefficients = normalized_linear_coefficients
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,
            num_points=num_points,
            dt=1.0,
            linear_coefficients=normalized_linear_coefficients,
        )


class DifficultyLinearStepper(NormalizedLinearStepper):
    linear_difficulties: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (0.0, -2.0),
    ):
        """
        Timestepper for d-dimensional (`d ∈ {1, 2, 3}`) linear PDEs on periodic
        boundary conditions with normalized dynamics in a difficulty-based
        interface.

        Different to `NormalizedLinearStepper`, the dynamics are defined by
        difficulties. The difficulties are a different combination of normalized
        dynamics, `num_spatial_dims`, and `num_points`.

            γᵢ = αᵢ Nⁱ 2ⁱ⁻¹ d

        with `d` the number of spatial dimensions, `N` the number of points, and
        `αᵢ` the normalized coefficient.

        This interface is more natural because the difficulties for all orders
        (given by `i`) are around 1.0. Additionally, they relate to stability
        condition of explicit Finite Difference schemes for the particular
        equations. For example, for advection (`i=1`), the absolute of the
        difficulty is the Courant-Friedrichs-Lewy (CFL) number.

        In the default configuration of this timestepper, the PDE is an
        advection equation with CFL number 2 solved in 1d with 48 resolution
        points to discretize the domain.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`. Default is
            1.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`. Default is 48.
        - `difficulties`: The difficulties of the normalized dynamics. This must
            be a tuple of floats. The length of the tuple defines the highest
            occuring linear derivative in the PDE. Default is `(0.0, -2.0)`.
        """
        self.linear_difficulties = linear_difficulties
        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            linear_difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_linear_coefficients=normalized_coefficients,
        )


class DiffultyLinearStepperSimple(DifficultyLinearStepper):
    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        difficulty: float = -2.0,
        order: int = 1,
    ):
        """
        A simple interface for `DifficultyLinearStepper` with only one
        difficulty and a given order.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`. Default is
            1.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`. Default is 48.
        - `difficulty`: The difficulty of the normalized dynamics. This must be
            a float. Default is -2.0.
        - `order`: The order of the derivative associated with the provided
            difficulty. The default of 1 is the advection equation.
        """
        difficulties = (0.0,) * (order) + (difficulty,)
        super().__init__(
            linear_difficulties=difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
