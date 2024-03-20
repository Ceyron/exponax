import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import GradientNormNonlinearFun


class GeneralGradientNormStepper(BaseStepper):
    coefficients: tuple[float, ...]
    gradient_norm_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        coefficients: tuple[float, ...] = (0.0, 0.0, -1.0, 0.0, -1.0),
        gradient_norm_scale: float = 1.0,
        order=2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for d-dimensional (`d ∈ {1, 2, 3}`) semi-linear PDEs
        consisting of a gradient norm nonlinearity and an arbitrary combination
        of (isotropic) linear operators.

        In 1d, the equation is given by

        ```
            uₜ + b₂ 1/2 (uₓ)² = sum_j a_j uₓˢ
        ```

        with `b₂` the gradient norm coefficient and `a_j` the coefficients of
        the linear operators. `uₓˢ` denotes the s-th derivative of `u` with
        respect to `x`. Oftentimes `b₂ = 1`.

        The number of channels is always one, no matter the number of spatial
        dimensions. The higher dimensional equation reads

        ```
            uₜ + b₂ 1/2 ‖ ∇u ‖₂² = sum_j a_j (1⋅∇ʲ)u
        ```

        The default configuration coincides with a Kuramoto-Sivashinsky equation
        in combustion format. Note that this requires negative values (because
        the KS usually defines their linear operators on the left hand side of
        the equation)

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
                corresponding to the derivatives. The length of this tuple
                represents the highest occuring derivative. The default value
                `(0.0, 0.0, -1.0, 0.0, -1.0)` corresponds to the Kuramoto-
                Sivashinsky equation in combustion format.
            - `gradient_norm_scale` (keyword-only): The scale of the gradient
                norm term `b₂`. Default: 1.0.
            - `order`: The order of the Exponential Time Differencing Runge
                Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0`
                only solves the linear part of the equation. Use higher values
                for higher accuracy and stability. The default choice of `2` is
                a good compromise for single precision floats.
            - `dealiasing_fraction`: The fraction of the wavenumbers to keep
                before evaluating the nonlinearity. The default 2/3 corresponds
                to Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2.
                Default: 2/3.
            - `num_circle_points`: How many points to use in the complex contour
                integral method to compute the coefficients of the exponential
                time differencing Runge Kutta method. Default: 16.
            - `circle_radius`: The radius of the contour used to compute the
                coefficients of the exponential time differencing Runge Kutta
                method. Default: 1.0.
        """
        self.coefficients = coefficients
        self.gradient_norm_scale = gradient_norm_scale
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
    ) -> GradientNormNonlinearFun:
        return GradientNormNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.gradient_norm_scale,
            zero_mode_fix=True,  # Todo: check this
        )
