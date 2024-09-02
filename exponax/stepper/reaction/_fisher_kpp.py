from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ..._spectral import build_laplace_operator
from ...nonlin_fun import PolynomialNonlinearFun


class FisherKPP(BaseStepper):
    diffusivity: float
    reactivity: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
        reactivity=1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Fisher-KPP equation
        on periodic boundary conditions. This reaction-diffusion equation is
        related to logistic growth and describes the spread of a population.

        In 1d, the Fisher-KPP equation is given by

        ```
            uₜ = ν uₓₓ + r u (1 - u)
        ```

        with `ν` the diffusivity and `r` the reactivity. In 1d, the state `u`
        has only one channel. As such the discretized state is represented by a
        tensor of shape `(1, num_points)`. For higher dimensions, the number of
        channels will be constant 1, no matter the dimension. The
        higher-dimensional equation reads

        ```
            uₜ = ν Δu + r u (1 - u)
        ```

        with `Δ` the Laplacian.

        The dynamics requires initial conditions in the range `[0, 1]`. Then,
        the expected temporal behavior is a collective spread and growth. The
        limit of the solution is the constant state `1`.

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
        - `diffusivity`: The diffusivity `ν`.
        - `reactivity`: The reactivity `r`.
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

        - The dynamics require initial conditions in the range `[0, 1]`.
            This can be achieved by combining any of the available IC generators
            with the [`exponax.ic.ClampingICGenerator`]. Alternatively, a good
            choice is also the [`exponax.ic.GaussianBlobs`]

        **Good Values:**

        - Use the `ClampingICGenerator` on `RandomTruncatedFourierSeries`
            with limits `[0, 1]` to generate initial conditions. Set
            `domain_extent = 1.0`, `num_points = 100`, `dt = 0.001`, and produce
            a trajectory of 500 steps. The final state of almost constant `1`
            will be reached after 200-400 steps.
        """
        self.dealiasing_fraction = dealiasing_fraction
        self.diffusivity = diffusivity
        self.reactivity = reactivity
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
        linear_operator = self.diffusivity * laplace + self.reactivity
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> PolynomialNonlinearFun:
        return PolynomialNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            dealiasing_fraction=self.dealiasing_fraction,
            coefficients=[0.0, 0.0, -self.reactivity],
        )
