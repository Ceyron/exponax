from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ..._spectral import build_laplace_operator
from ...nonlin_fun import PolynomialNonlinearFun


class SwiftHohenberg(BaseStepper):
    reactivity: float
    critical_number: float
    polynomial_coefficients: tuple[float, ...]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        reactivity: float = 0.7,
        critical_number: float = 1.0,
        polynomial_coefficients: tuple[float, ...] = (0.0, 0.0, 1.0, -1.0),
        order: int = 2,
        # Needs lower value due to cubic nonlinearity
        dealiasing_fraction: float = 1 / 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Swift-Hohenberg
        reaction-diffusion equation on periodic boundary conditions (works best
        in 2d). This reaction-diffusion equation is a model for pattern
        formation, for example, the fingerprints on a human finger.

        In 1d, the Swift-Hohenberg equation is given by

        ```
            uₜ = r u - (k + ∂ₓₓ)² u + g(u)
        ```

        with `r` the reactivity, `k` the critical number, `∂ₓₓ` the second
        derivative operator. `g(u)` can be any smooth function. This equation
        restricts to the case of polynomial functions, i.e.

        ```
            g(u) = ∑ᵢ cᵢ uⁱ
        ```

        with `cᵢ` the polynomial coefficients.

        The state only has one channel, no matter the spatial dimension. The
        higher dimensional generarlization reads

        ```
            uₜ = r u - (k + Δ)² u + g(u)
        ```

        with `Δ` the Laplacian. Since the Laplacian is squared, there will be
        spatial mixing.

        The expected temporal behavior is a collective pattern formation which
        will be attained in a steady state.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `reactivity`: The reactivity `r`. Default is `0.7`.
        - `critical_number`: The critical number `k`. Default is `1.0`.
        - `polynomial_coefficients`: The coefficients `cᵢ` of the polynomial
            function `g(u)`. Default is `(0.0, 0.0, 1.0, -1.0)`. This refers to
            a polynomial of `u² - u³`.
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
        """
        self.reactivity = reactivity
        self.critical_number = critical_number
        self.polynomial_coefficients = polynomial_coefficients
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
        linear_operator = self.reactivity - (self.critical_number + laplace) ** 2
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> PolynomialNonlinearFun:
        return PolynomialNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            dealiasing_fraction=self.dealiasing_fraction,
            coefficients=self.polynomial_coefficients,
        )
