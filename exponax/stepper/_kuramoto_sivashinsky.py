from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import ConvectionNonlinearFun, GradientNormNonlinearFun


class KuramotoSivashinsky(BaseStepper):
    gradient_norm_scale: float
    second_order_scale: float
    fourth_order_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        gradient_norm_scale: float = 1.0,
        second_order_scale: float = 1.0,
        fourth_order_scale: float = 1.0,
        dealiasing_fraction: float = 2 / 3,
        order: int = 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Kuramoto-Sivashinsky
        equation on periodic boundary conditions. Uses the **combustion format**
        (or non-conservative format). Most deep learning papers in 1d considered
        the conservative format available as
        [`exponax.stepper.KuramotoSivashinskyConservative`][].

        In 1d, the KS equation is given by

        ```
            uₜ + b₂ 1/2 (uₓ)² + ψ₁ uₓₓ + ψ₂ uₓₓₓₓ = 0
        ```

        with `b₂` the gradient-norm coefficient, `ψ₁` the second-order scale and
        `ψ₂` the fourth-order. If the latter two terms were on the right-hand
        side, they could be interpreted as diffusivity and hyper-diffusivity,
        respectively. Here, the second-order term acts destabilizing (increases
        the energy of the system) and the fourth-order term acts stabilizing
        (decreases the energy of the system). A common configuration is `b₂ = ψ₁
        = ψ₂ = 1` and the dynamics are only adapted using the `domain_extent`.
        For this, we espect the KS equation to experience spatio-temporal chaos
        roughly once `L > 60`.

        In this combustion (=non-conservative) format, the number of channels
        does **not** grow with the spatial dimension. A 2d KS still only has a
        single channel. In higher dimensions, the equation reads

        ```
            uₜ + b₂ 1/2 ‖ ∇u ‖₂² + ψ₁ν (∇ ⋅ ∇) u + ψ₂ ((∇ ⊗ ∇) ⋅ (∇ ⊗ ∇))u = 0
        ```

        with `‖ ∇u ‖₂` the gradient norm, `∇ ⋅ ∇` effectively is the Laplace
        operator `Δ`. The fourth-order term generalizes to `((∇ ⊗ ∇) ⋅ (∇ ⊗ ∇))`
        which is **not** the same as `ΔΔ = Δ²` since the latter would mix
        spatially.

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
        - `gradient_norm_scale`: The gradient-norm coefficient `b₂`. Note
            that the gradient norm is already scaled by 1/2. This factor allows
            for further modification. Default: 1.0.
        - `second_order_scale`: The "diffusivity" `ψ₁` in the KS equation.
        - `fourth_order_diffusivity`: The "hyper-diffusivity" `ψ₂` in the KS
            equation.
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

        - The KS equation enters a chaotic state if the domain extent is
            chosen large enough. In this chaotic attractor it can run
            indefinitely. It is in balancing state of the second-order term
            producing new energy, the nonlinearity transporting it into higher
            modes where the fourth-order term dissipates it.
        - If the domain extent is chosen large enough to eventually enter a
            chaotic state, the initial condition does not really matter. Since
            the KS "produces its own energy", the energy spectrum for the
            chaotic attractor is independent of the initial condition.
        - However, since the KS develops a certain spectrum based on the
            domain length, make sure to use enough discretization point to
            capture the highes occuring mode. For a domain extent of 60, this
            requires at least roughly 100 `num_points` in single precision
            floats.
        - For domain lengths smaller than the threshold to enter chaos, the
            KS equation, exhibits various other patterns like propagating waves,
            etc.
        - For higher dimensions (i.e., `num_spatial_dims > 1`), a chaotic
            state is already entered for smaller domain extents. For more
            details and the kind of dynamics that can occur see:
            https://royalsocietypublishing.org/doi/10.1098/rspa.2014.0932

        **Good Values:**

        - For a simple spatio-temporal chaos in 1d, set
            `num_spatial_dims=1`, `domain_extent=60`, `num_points=100`,
            `dt=0.1`. The initial condition can be anything, important is that
            it is mean zero. The first 200-500 steps of the trajectory will be
            the transitional phase, after that the chaotic attractor is reached.
        """
        self.gradient_norm_scale = gradient_norm_scale
        self.second_order_scale = second_order_scale
        self.fourth_order_scale = fourth_order_scale
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
        # Minuses are required to move the terms to the right-hand side
        linear_operator = -self.second_order_scale * build_laplace_operator(
            derivative_operator, order=2
        ) - self.fourth_order_scale * build_laplace_operator(
            derivative_operator, order=4
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
            zero_mode_fix=True,
            scale=self.gradient_norm_scale,
        )


class KuramotoSivashinskyConservative(BaseStepper):
    convection_scale: float
    second_order_scale: float
    fourth_order_scale: float
    single_channel: bool
    conservative: bool
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        convection_scale: float = 1.0,
        second_order_scale: float = 1.0,
        fourth_order_scale: float = 1.0,
        single_channel: bool = False,
        conservative: bool = True,
        dealiasing_fraction: float = 2 / 3,
        order: int = 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Using the fluid dynamics form of the KS equation (i.e. similar to the
        Burgers equation).

        In 1d, the KS equation is given by

        ```
            uₜ + b₁ 1/2 (u²)ₓ + ψ₁ uₓₓ + ψ₂ uₓₓₓₓ = 0
        ```

        with `b₁` the convection coefficient, `ψ₁` the second-order scale and
        `ψ₂` the fourth-order. If the latter two terms were on the right-hand
        side, they could be interpreted as diffusivity and hyper-diffusivity,
        respectively. Here, the second-order term acts destabilizing (increases
        the energy of the system) and the fourth-order term acts stabilizing
        (decreases the energy of the system). A common configuration is `b₁ = ψ₁
        = ψ₂ = 1` and the dynamics are only adapted using the `domain_extent`.
        For this, we espect the KS equation to experience spatio-temporal chaos
        roughly once `L > 60`.

        !!! info
            In this fluid dynamics (=conservative) format, the number of
            channels grows with the spatial dimension. However, it seems that
            this format does not generalize well to higher dimensions. For
            higher dimensions, consider using the combustion format
            (`exponax.stepper.KuramotoSivashinsky`) instead.

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
        - `convection_scale`: The convection coefficient `b₁`. Note that the
            convection term is already scaled by 1/2. This factor allows for
            further modification. Default: 1.0.
        - `second_order_scale`: The "diffusivity" `ψ₁` in the KS equation.
        - `fourth_order_diffusivity`: The "hyper-diffusivity" `ψ₂` in the KS
            equation.
        - `single_channel`: Whether to use a single channel for the spatial
            dimension. Default: `False`.
        - `conservative`: Whether to use the conservative format. Default:
            `True`.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. The default 2/3 corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2. Default:
            2/3.
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
        self.convection_scale = convection_scale
        self.second_order_scale = second_order_scale
        self.fourth_order_scale = fourth_order_scale
        self.single_channel = single_channel
        self.conservative = conservative
        self.dealiasing_fraction = dealiasing_fraction

        if num_spatial_dims > 1:
            print(
                "Warning: The KS equation in conservative format does not generalize well to higher dimensions."
            )
            print(
                "Consider using the combustion format (`exponax.stepper.KuramotoSivashinsky`) instead."
            )

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
        # Minuses are required to move the terms to the right-hand side
        linear_operator = -self.second_order_scale * build_laplace_operator(
            derivative_operator, order=2
        ) - self.fourth_order_scale * build_laplace_operator(
            derivative_operator, order=4
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
            conservative=self.conservative,
        )
