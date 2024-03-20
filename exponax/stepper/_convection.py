import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import ConvectionNonlinearFun


class GeneralConvectionStepper(BaseStepper):
    coefficients: tuple[float, ...]
    convection_scale: float
    dealiasing_fraction: float
    single_channel: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        coefficients: tuple[float, ...] = (0.0, 0.0, 0.01),
        convection_scale: float = 1.0,
        single_channel: bool = False,
        order=2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) semi-linear PDEs
        consisting of a convection nonlinearity and an arbitrary combination of
        (isotropic) linear derivatives.

        In 1d, the equation is given by

        ```
            uₜ + b₁ 1/2 (u²)ₓ = sum_j a_j uₓˢ

        ```

        with `b₁` the convection coefficient and `a_j` the coefficients of the
        linear operators. `uₓˢ` denotes the s-th derivative of `u` with respect
        to `x`. Oftentimes `b₁ = 1`.

        In the default configuration, the number of channel grows with the
        number of spatial dimensions. The higher dimensional equation reads

        ```
            uₜ + b₁ 1/2 ∇ ⋅ (u ⊗ u) = sum_j a_j (1⋅∇ʲ)u
        ```

        Alternatively, with `single_channel=True`, the number of channels can be
        kept to constant 1 no matter the number of spatial dimensions.

        Depending on the collection of linear coefficients can be represented,
        for example:
            - Burgers equation with `a = (0, 0, 0.01)` with `len(a) = 3`
            - KdV equation with `a = (0, 0, 0, 0.01)` with `len(a) = 4`

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
                `(0.0, 0.0, 0.01)` corresponds to the Burgers equation (because
                of the diffusion)
            - `convection_scale` (keyword-only): The scale `b₁` of the
                convection term. Default is `1.0`.
            - `single_channel`: Whether to use the single channel mode in higher
                dimensions. In this case the the convection is `b₁ (∇ ⋅ 1)(u²)`.
                In this case, the state always has a single channel, no matter
                the spatial dimension. Default: False.
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
        self.convection_scale = convection_scale
        self.single_channel = single_channel
        self.dealiasing_fraction = dealiasing_fraction

        if single_channel:
            num_channels = 1
        else:
            # number of channels grow with the spatial dimension
            num_channels = num_spatial_dims

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=num_channels,
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
    ) -> ConvectionNonlinearFun:
        return ConvectionNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.convection_scale,
            single_channel=self.single_channel,
        )
