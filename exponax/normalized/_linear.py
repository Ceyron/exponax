import jax.numpy as jnp
from jaxtyping import Array

from .._base_stepper import BaseStepper
from ..nonlin_fun import ZeroNonlinearFun
from ._utils import extract_normalized_coefficients_from_difficulty


class NormalizedLinearStepper(BaseStepper):
    normalized_coefficients: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients: tuple[float, ...] = (0.0, -0.5, 0.01),
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
        self.normalized_coefficients = normalized_coefficients
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(self, derivative_operator: Array) -> Array:
        linear_operator = sum(
            jnp.sum(
                c * (derivative_operator) ** i,
                axis=0,
                keepdims=True,
            )
            for i, c in enumerate(self.normalized_coefficients)
        )
        return linear_operator

    def _build_nonlinear_fun(self, derivative_operator: Array):
        return ZeroNonlinearFun(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
        )


class DifficultyLinearStepper(NormalizedLinearStepper):
    difficulties: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        difficulties: tuple[float, ...] = (0.0, -2.0),
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
        self.difficulties = difficulties
        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_coefficients=normalized_coefficients,
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
            difficulties=difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
