from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ..._spectral import build_laplace_operator
from ...nonlin_fun import PolynomialNonlinearFun


class AllenCahn(BaseStepper):
    diffusivity: float
    first_order_coefficient: float
    third_order_coefficient: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 5e-3,
        first_order_coefficient: float = 1.0,
        third_order_coefficient: float = -1.0,
        order: int = 2,
        # Needs lower value due to cubic nonlinearity
        dealiasing_fraction: float = 1 / 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Allen-Cahn
        reaction-diffusion equation on periodic boundary conditions. This
        reaction-diffusion equation is a model for phase separation, for example
        the separation of oil and water.

        In 1d, the Allen-Cahn equation is given by

        ```
            uₜ = ν uₓₓ + c₁ u + c₃ u³
        ```

        with `ν` the diffusivity, `c₁` the first order coefficient, and `c₃` the
        third order coefficient. No matter the spatial dimension, the state
        always only has one channel. In higher dimensions, the equation reads

        ```
            uₜ = ν Δu + c₁ u + c₃ u³
        ```

        with `Δ` the Laplacian.

        The expected temporal behavior is the formation of sharp interfaces
        between the two phases. The limit of the solution is a step function
        that separates the two phases.

        Note that the Allen-Cahn is often solved with Dirichlet boundary
        conditions, but here we use periodic boundary conditions.

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
        - `diffusivity`: The diffusivity `ν`. The default value is `5e-3`.
        - `first_order_coefficient`: The first order coefficient `c₁`. The
            default value is `1.0`.
        - `third_order_coefficient`: The third order coefficient `c₃`. The
            default value is `-1.0`.
        - `dealiasing_fraction`: The fraction of the highest wavenumbers to
            dealias. Default is `1/2` because the default polynomial has a
            highest degree of 3.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for higher
            accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.

        **Notes:**

        - See
            https://github.com/chebfun/chebfun/blob/db207bc9f48278ca4def15bf90591bfa44d0801d/spin.m#L48
            for an example IC of the Allen-Cahn in 1d.
        """
        self.diffusivity = diffusivity
        self.first_order_coefficient = first_order_coefficient
        self.third_order_coefficient = third_order_coefficient
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        laplace = build_laplace_operator(derivative_operator, order=2)
        linear_operator = self.diffusivity * laplace + self.first_order_coefficient
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> PolynomialNonlinearFun:
        return PolynomialNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            dealiasing_fraction=self.dealiasing_fraction,
            coefficients=[0.0, 0.0, 0.0, self.third_order_coefficient],
        )
