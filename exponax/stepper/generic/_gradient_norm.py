import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ...nonlin_fun import GradientNormNonlinearFun
from ._utils import (
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_gradient_norm_scale_from_difficulty,
)


class GeneralGradientNormStepper(BaseStepper):
    linear_coefficients: tuple[float, ...]
    gradient_norm_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        linear_coefficients: tuple[float, ...] = (0.0, 0.0, -1.0, 0.0, -1.0),
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
        in combustion format (see `exponax.stepper.KuramotoSivashinsky`). Note
        that this requires negative values (because the KS usually defines their
        linear operators on the left hand side of the equation)

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
            corresponding to the derivatives. The length of this tuple
            represents the highest occuring derivative. The default value `(0.0,
            0.0, -1.0, 0.0, -1.0)` corresponds to the Kuramoto- Sivashinsky
            equation in combustion format.
        - `gradient_norm_scale` (keyword-only): The scale of the gradient
            norm term `b₂`. Default: 1.0.
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
        """
        self.linear_coefficients = linear_coefficients
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
            for i, c in enumerate(self.linear_coefficients)
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


class NormalizedGradientNormStepper(GeneralGradientNormStepper):
    normalized_linear_coefficients: tuple[float, ...]
    normalized_gradient_norm_scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_linear_coefficients: tuple[float, ...] = (
            0.0,
            0.0,
            -1.0 * 0.1 / (60.0**2),
            0.0,
            -1.0 * 0.1 / (60.0**4),
        ),
        normalized_gradient_norm_scale: float = 1.0 * 0.1 / (60.0**2),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the **normalized** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of a gradient norm nonlinearity and an
        arbitrary combination of (isotropic) linear operators. Uses a normalized
        interface, i.e., the domain is scaled to `Ω = (0, 1)ᵈ` and time step
        size is `Δt = 1.0`.

        See `exponax.stepper.generic.GeneralGradientNormStepper` for more
        details on the functional form of the PDE.

        The number of channels do **not** grow with the number of spatial
        dimensions. They are always one.

        Under the default settings, it behaves like the Kuramoto-Sivashinsky
        equation in combustion format under the following settings.

        By default: the KS equation on L=60.0

        ```python

        exponax.stepper.KuramotoSivashinsky(
            num_spatial_dims=D, domain_extent=60.0, num_points=N, dt=0.1,
            gradient_norm_scale=1.0, second_order_diffusivity=1.0,
            fourth_order_diffusivity=1.0,
        )
        ```

        Note that the coefficient list requires a negative sign because the
        linear derivatives are moved to the right-hand side in this generic
        interface.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `normalized_coefficients`: The list of coefficients `a_j`
            corresponding to the derivatives. The length of this tuple
            represents the highest occuring derivative. The default value `(0.0,
            0.0, -1.0 * 0.1 / (60.0**2), 0.0, -1.0 * 0.1 / (60.0**4))`
            corresponds to the Kuramoto-Sivashinsky equation in combustion
            format on a domain of size `L=60.0` with a time step size of
            `Δt=0.1`.
        - `normalized_gradient_norm_scale`: The scale of the gradient
            norm term `b₂`. Default: `1.0 * 0.1 / (60.0**2)`.
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
        """
        self.normalized_linear_coefficients = normalized_linear_coefficients
        self.normalized_gradient_norm_scale = normalized_gradient_norm_scale
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,
            num_points=num_points,
            dt=1.0,
            linear_coefficients=normalized_linear_coefficients,
            gradient_norm_scale=normalized_gradient_norm_scale,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )


class DifficultyGradientNormStepper(NormalizedGradientNormStepper):
    linear_difficulties: tuple[float, ...]
    gradient_norm_difficulty: float

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (0.0, 0.0, -0.128, 0.0, -0.32768),
        gradient_norm_difficulty: float = 0.064,
        maximum_absolute: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the **difficulty-based** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of a gradient norm nonlinearity and an
        arbitrary combination of (isotropic) linear operators. Uses a
        difficulty-based interface where the "intensity" of the dynamics reduces
        with increasing resolution. This is intended such that emulator learning
        problems on two resolutions are comparibly difficult.

        Different to `exponax.stepper.generic.NormalizedGradientNormStepper`,
        the dynamics are defined by difficulties. The difficulties are a
        different combination of normalized dynamics, `num_spatial_dims`, and
        `num_points`.

            γᵢ = αᵢ Nⁱ 2ⁱ⁻¹ d

        with `d` the number of spatial dimensions, `N` the number of points, and
        `αᵢ` the normalized coefficient.

        The difficulty of the nonlinear convection scale is defined by

            δ₂ = β₂ * M * N² * D

        with `M` the maximum absolute value of the input state (typically `1.0`
        if one uses the `exponax.ic` random generators with the `max_one=True`
        argument).

        This interface is more natural than the normalized interface because the
        difficulties for all orders (given by `i`) are around 1.0. Additionally,
        they relate to stability condition of explicit Finite Difference schemes
        for the particular equations. For example, for advection (`i=1`), the
        absolute of the difficulty is the Courant-Friedrichs-Lewy (CFL) number.

        Under the default settings, this timestepper represents the
        Kuramoto-Sivashinsky equation (in combustion format).

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `linear_difficulties`: The list of difficulties `γᵢ` corresponding to
            the derivatives. The length of this tuple represents the highest
            occuring derivative. The default value `(0.0, 0.0, -0.128, 0.0,
            -0.32768)` corresponds to the Kuramoto-Sivashinsky equation in
            combustion format (because it contains both a negative diffusion and
            a negative hyperdiffusion term).
        - `gradient_norm_difficulty`: The difficulty of the gradient norm term
            `δ₂`.
        - `maximum_absolute`: The maximum absolute value of the input state. This
            is used to scale the gradient norm term.
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
        """
        self.linear_difficulties = linear_difficulties
        self.gradient_norm_difficulty = gradient_norm_difficulty

        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            linear_difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
        normalized_gradient_norm_scale = (
            extract_normalized_gradient_norm_scale_from_difficulty(
                gradient_norm_difficulty,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
                maximum_absolute=maximum_absolute,
            )
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_linear_coefficients=normalized_coefficients,
            normalized_gradient_norm_scale=normalized_gradient_norm_scale,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )
