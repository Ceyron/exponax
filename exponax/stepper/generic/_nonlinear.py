import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ...nonlin_fun import GeneralNonlinearFun
from ._utils import (
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_nonlinear_scales_from_difficulty,
)


class GeneralNonlinearStepper(BaseStepper):
    linear_coefficients: tuple[float, ...]
    nonlinear_coefficients: tuple[float, float, float]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        linear_coefficients: tuple[float, ...] = (0.0, 0.0, 0.01),
        nonlinear_coefficients: tuple[float, float, float] = (0.0, -1.0, 0.0),
        order=2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for d-dimensional (`d ∈ {1, 2, 3}`) semi-linear PDEs
        consisting of a quadratic, a single-channel convection, and a gradient
        norm nonlinearity together with an arbitrary combination of (isotropic)
        linear derivatives.

        In 1d, the PDE is of the form

        ```
            uₜ = b₀ u² + b₁ 1/2 (u²)ₓ + b₂ 1/2 (uₓ)² + sum_j a_j uₓʲ
        ```

        where `b₀`, `b₁`, `b₂` are the coefficients of the quadratic,
        convection, and gradient norm nonlinearity, respectively, and `a_j` are
        the coefficients of the linear derivatives. Effectively, this
        timestepper is a combination of the
        `exponax.stepper.generic.GeneralPolynomialStepper` (with only the
        coefficient to the quadratic polynomial being set with `b₀`), the
        `exponax.stepper.generic.GeneralConvectionStepper` (with the
        single-channel hack activated via `single_channel=True` and the
        convection scale set with `- b₁`), and the
        `exponax.stepper.generic.GeneralGradientNormStepper` (with the gradient
        norm scale set with `- b₂`).

        !!! warning
            In contrast to the
            `exponax.stepper.generic.GeneralConvectionStepper` and the
            `exponax.stepper.generic.GeneralGradientNormStepper`, the nonlinear
            terms are no considered to be on right-hand side of the PDE. Hence,
            in order to get the same dynamics as in the other steppers, the
            coefficients must be negated. (This is not relevant for the
            coefficient of the quadratic polynomial because in the
            `exponax.stepper.generic.GeneralPolynomialStepper` the polynomial
            nonlinearity is already on the right-hand side.)

        The higher-dimensional generalization is

        ```
            uₜ = b₀ u² + b₁ 1/2 (1⃗ ⋅ ∇)(u²) + b₂ 1/2 ‖ ∇u ‖₂² + sum_j a_j uₓˢ
        ```

        Under the default configuration, this correspons to a Burgers equation
        in single-channel mode.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions the
            domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `linear_coefficients`: The list of coefficients `a_j` corresponding to
            the derivatives. The length of this tuple represents the highest
            occuring derivative. The default value `(0.0, 0.0, 0.01)` together
            with the default `nonlinear_coefficients` corresponds to the Burgers
            equation.
        - `nonlinear_coefficients`: The list of coefficients `b₀`, `b₁`, `b₂`
            (in this order). The default value `(0.0, -1.0, 0.0)` corresponds to
            a (single-channel) convection nonlinearity scaled with `1.0`. Note
            that all nonlinear contributions are considered to be on the
            right-hand side of the PDE. Hence, in order to get the "correct
            convection" dynamics, the coefficients must be negated.
        - `order`: The order of the ETDRK method to use. Must be one of {0, 1, 2,
            3, 4}. The option `0` only solves the linear part of the equation.
            Use higher values for higher accuracy and stability. The default
            choice of `2` is a good compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep before
            evaluating the nonlinearity. The default value `2/3` corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta method.
        """
        if len(nonlinear_coefficients) != 3:
            raise ValueError(
                "The nonlinear coefficients list must have exactly 3 elements"
            )
        self.linear_coefficients = linear_coefficients
        self.nonlinear_coefficients = nonlinear_coefficients
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
    ) -> GeneralNonlinearFun:
        return GeneralNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale_list=self.nonlinear_coefficients,
            zero_mode_fix=True,  # ToDo: check this
        )


class NormalizedNonlinearStepper(GeneralNonlinearStepper):
    normalized_linear_coefficients: tuple[float, ...]
    normalized_nonlinear_coefficients: tuple[float, float, float]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_linear_coefficients: tuple[float, ...] = (0.0, 0.0, 0.1 * 0.1),
        normalized_nonlinear_coefficients: tuple[float, float, float] = (
            0.0,
            -1.0 * 0.1,
            0.0,
        ),
        order=2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default Burgers.

        Timesteppr for **normalized** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of a quadratic, a single-channel convection,
        and a gradient norm nonlinearity together with an arbitrary combination
        of (isotropic) linear derivatives. Uses a normalized interface, i.e.,
        the domain is scaled to `Ω = (0, 1)ᵈ` and time step size is `Δt = 1.0`.

        See `exponax.stepper.generic.GeneralNonlinearStepper` for more details
        on the functional form of the PDE.

        Behaves like a single-channel Burgers equation by default under the
        following settings

        ```python

        exponax.stepper.Burgers(
            num_spatial_dims=num_spatial_dims, domain_extent=1.0,
            num_points=num_points, dt=1.0, convection_scale=1.0,
            diffusivity=0.1, single_channel=True,
        )
        ```

        Effectively, this timestepper is a combination of the
        `exponax.stepper.generic.NormalizedPolynomialStepper` (with only the
        coefficient to the quadratic polynomial being set with `b₀`), the
        `exponax.stepper.generic.NormalizedConvectionStepper` (with the
        single-channel hack activated via `single_channel=True` and the
        convection scale set with `- b₁`), and the
        `exponax.stepper.generic.NormalizedGradientNormStepper` (with the
        gradient norm scale set with `- b₂`).

        !!! warning
            In contrast to the
            `exponax.stepper.generic.NormalizedConvectionStepper` and the
            `exponax.stepper.generic.NormalizedGradientNormStepper`, the
            nonlinear terms are no considered to be on right-hand side of the
            PDE. Hence, in order to get the same dynamics as in the other
            steppers, the coefficients must be negated. (This is not relevant
            for the coefficient of the quadratic polynomial because in the
            `exponax.stepper.generic.NormalizedPolynomialStepper` the polynomial
            nonlinearity is already on the right-hand side.)


        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`.
        - `normalized_linear_coefficients`: The list of coefficients `αⱼ`
            corresponding to the linear derivatives. The length of this tuple
            represents the highest occuring derivative. The default value `(0.0,
            0.0, 0.1 * 0.1)` together with the default
            `normalized_nonlinear_coefficients` corresponds to the Burgers
            equation (in single-channel mode).
        - `normalized_nonlinear_coefficients`: The list of coefficients `β₀`,
            `β₁`, and `β₂` (in this order) corresponding to the quadratic,
            (single-channel) convection, and gradient norm nonlinearity,
            respectively. The default value `(0.0, -1.0 * 0.1, 0.0)` corresponds
            to a (single-channel) convection nonlinearity scaled with `0.1`.
            Note that all nonlinear contributions are considered to be on the
            right-hand side of the PDE. Hence, in order to get the "correct
            convection" dynamics, the coefficients must be negated. (Also
            relevant for the gradient norm nonlinearity)
        - `order`: The order of the ETDRK method to use. Must be one of {0, 1, 2,
            3, 4}. The option `0` only solves the linear part of the equation.
            Use higher values for higher accuracy and stability. The default
            choice of `2` is a good compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep before
            evaluating the nonlinearity. The default value `2/3` corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta method.
        """

        self.normalized_linear_coefficients = normalized_linear_coefficients
        self.normalized_nonlinear_coefficients = normalized_nonlinear_coefficients

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
            linear_coefficients=normalized_linear_coefficients,
            nonlinear_coefficients=normalized_nonlinear_coefficients,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )


class DifficultyNonlinearStepper(NormalizedNonlinearStepper):
    linear_difficulties: tuple[float, ...]
    nonlinear_difficulties: tuple[float, float, float]

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (
            0.0,
            0.0,
            0.1 * 0.1 / 1.0 * 48**2 * 2,
        ),
        nonlinear_difficulties: tuple[float, float, float] = (
            0.0,
            -1.0 * 0.1 / 1.0 * 48,
            0.0,
        ),
        maximum_absolute: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for **difficulty-based** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of a quadratic, a single-channel convection,
        and a gradient norm nonlinearity together with an arbitrary combination
        of (isotropic) linear derivatives. Uses a difficulty-based interface
        where the "intensity" of the dynamics reduces with increasing
        resolution. This is intended such that emulator learning problems on two
        resolutions are comparibly difficult.

        Different to `exponax.stepper.generic.NormalizedNonlinearStepper`, the
        dynamics are defined by difficulties. The difficulties are a different
        combination of normalized dynamics, `num_spatial_dims`, and
        `num_points`.

            γᵢ = αᵢ Nⁱ 2ⁱ⁻¹ d

        with `d` the number of spatial dimensions, `N` the number of points, and
        `αᵢ` the normalized coefficient.

        The difficulties of the nonlinear terms are

            δ₀ = β₀

            δ₁ = β₁ * M * N * D

            δ₂ = β₂ * M * N² * D

        with `βᵢ` the normalized coefficient and `M` the maximum absolute value
        of the input state (typically `1.0` if one uses the `exponax.ic` random
        generators with the `max_one=True` argument).

        This interface is more natural than the normalized interface because the
        difficulties for all orders (given by `i`) are around 1.0. Additionally,
        they relate to stability condition of explicit Finite Difference schemes
        for the particular equations. For example, for advection (`i=1`), the
        absolute of the difficulty is the Courant-Friedrichs-Lewy (CFL) number.

        Under the default settings, this timestep corresponds to a Burgers
        equation in single-channel mode.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`.
        - `linear_difficulties`: The list of difficulties `γᵢ` corresponding to
            the linear derivatives. The length of this tuple represents the
            highest occuring derivative. The default value `(0.0, 0.0, 0.1 * 0.1
            / 1.0 * 48**2 * 2)` together with the default `nonlinear_difficulties`
            corresponds to the Burgers equation.
        - `nonlinear_difficulties`: The list of difficulties `δ₀`, `δ₁`, and `δ₂`
            (in this order) corresponding to the quadratic, (single-channel)
            convection, and gradient norm nonlinearity, respectively. The default
            value `(0.0, -1.0 * 0.1 / 1.0 * 48, 0.0)` corresponds to a
            (single-channel) convection nonlinearity. Note that all nonlinear
            contributions are considered to be on the right-hand side of the PDE.
        - `maximum_absolute`: The maximum absolute value of the input state. This
            is used to scale the nonlinear difficulties.
        - `order`: The order of the ETDRK method to use. Must be one of {0, 1, 2,
            3, 4}. The option `0` only solves the linear part of the equation.
            Use higher values for higher accuracy and stability. The default
            choice of `2` is a good compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep before
            evaluating the nonlinearity. The default value `2/3` corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta method.
        """
        self.linear_difficulties = linear_difficulties
        self.nonlinear_difficulties = nonlinear_difficulties

        normalized_coefficients_linear = (
            extract_normalized_coefficients_from_difficulty(
                linear_difficulties,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
            )
        )
        normalized_coefficients_nonlinear = (
            extract_normalized_nonlinear_scales_from_difficulty(
                nonlinear_difficulties,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
                maximum_absolute=maximum_absolute,
            )
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_linear_coefficients=normalized_coefficients_linear,
            normalized_nonlinear_coefficients=normalized_coefficients_nonlinear,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )
