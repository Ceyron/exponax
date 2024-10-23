import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ...nonlin_fun import PolynomialNonlinearFun
from ._utils import extract_normalized_coefficients_from_difficulty


class GeneralPolynomialStepper(BaseStepper):
    linear_coefficients: tuple[float, ...]
    polynomial_coefficients: tuple[float, ...]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        linear_coefficients: tuple[float, ...] = (10.0, 0.0, 1.0),
        polynomial_coefficients: tuple[float, ...] = (0.0, 0.0, -10.0),
        order=2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) semi-linear PDEs
        consisting of an arbitrary combination of polynomial nonlinearities and
        (isotropic) linear derivatives. This can be used to represent a wide
        array of reaction-diffusion equations.

        In 1d, the PDE is of the form

        ```
            uₜ = ∑ₖ pₖ uᵏ + ∑ⱼ aⱼ uₓʲ
        ```

        where `pₖ` are the polynomial coefficients and `aⱼ` are the linear
        coefficients. `uᵏ` denotes `u` pointwise raised to the power of `k`
        (hence the polynomial contribution) and `uₓʲ` denotes the `j`-th
        derivative of `u`.

        The higher-dimensional generalization reads

        ```
            uₜ = ∑ₖ pₖ uᵏ + ∑ⱼ a_j (1⋅∇ʲ)u

        ```

        where `∇ʲ` is the `j`-th derivative operator.

        The default configuration corresponds to the Fisher-KPP equation with
        the following settings

        ```python

        exponax.stepper.reaction.FisherKPP(
            num_spatial_dims=num_spatial_dims, domain_extent=domain_extent,
            num_points=num_points, dt=dt, diffusivity=0.01, reactivity=-10.0,
            #TODO: Check this
        )
        ```

        Note that the effect of polynomial_scale[1] is similar to the effect of
        coefficients[0] with the difference that in ETDRK integration the latter
        is treated anlytically and should be preferred.

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
        - `linear_coefficients`: The list of coefficients `a_j` corresponding to the
            derivatives. The length of this tuple represents the highest
            occuring derivative. The default value `(10.0, 0.0, 0.01)` in
            combination with the default `polynomial_coefficients` corresponds to the
            Fisher-KPP equation.
        - `polynomial_coefficients`: The list of scales `pₖ` corresponding to the
            polynomial contributions. The length of this tuple represents the
            highest occuring polynomial. The default value `(0.0, 0.0, 10.0)` in
            combination with the default `linear_coefficients` corresponds to the
            Fisher-KPP equation.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for higher
            accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. The default 2/3 corresponds to
            Orszag's 2/3 rule which is sufficient if the highest occuring
            polynomial is quadratic (i.e., there are at maximum three entries in
            the `polynomial_scales` tuple).
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method.
        """
        self.linear_coefficients = linear_coefficients
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
    ) -> PolynomialNonlinearFun:
        return PolynomialNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            dealiasing_fraction=self.dealiasing_fraction,
            coefficients=self.polynomial_coefficients,
        )


class NormalizedPolynomialStepper(GeneralPolynomialStepper):
    normalized_linear_coefficients: tuple[float, ...]
    normalized_polynomial_coefficients: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_linear_coefficients: tuple[float, ...] = (
            10.0 * 0.001 / (10.0**0),
            0.0,
            1.0 * 0.001 / (10.0**2),
        ),
        normalized_polynomial_coefficients: tuple[float, ...] = (
            0.0,
            0.0,
            -10.0 * 0.001,
        ),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the **normalized** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of an arbitrary combination of polynomial
        nonlinearities and (isotropic) linear derivatives. Uses a normalized
        interface, i.e., the domain is scaled to `Ω = (0, 1)ᵈ` and time step
        size is `Δt = 1.0`.

        See `exponax.stepper.generic.GeneralPolynomialStepper` for more details
        on the functional form of the PDE.

        The default settings correspond to the Fisher-KPP equation.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`.
        - `normalized_linear_coefficients`: The list of coefficients `α_j`
            corresponding to the derivatives. The length of this tuple
            represents the highest occuring derivative. The default value
            corresponds to the Fisher-KPP equation.
        - `normalized_polynomial_coefficients`: The list of scales `βₖ`
            corresponding to the polynomial contributions. The length of this
            tuple represents the highest occuring polynomial. The default value
            corresponds to the Fisher-KPP equation.
        - `order`: The order of the Exponential Time Differencing Runge Kutta
            method. Must be one of {0, 1, 2, 3, 4}. The option `0` only solves
            the linear part of the equation. Use higher values for higher
            accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep before
            evaluating the nonlinearity. The default 2/3 corresponds to Orszag's
            2/3 rule which is sufficient if the highest occuring polynomial is
            quadratic (i.e., there are at maximum three entries in the
            `normalized_polynomial_scales` tuple).
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method.
        """
        self.normalized_linear_coefficients = normalized_linear_coefficients
        self.normalized_polynomial_coefficients = normalized_polynomial_coefficients

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
            linear_coefficients=normalized_linear_coefficients,
            polynomial_coefficients=normalized_polynomial_coefficients,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )


class DifficultyPolynomialStepper(NormalizedPolynomialStepper):
    linear_difficulties: tuple[float, ...]
    polynomial_difficulties: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (
            10.0 * 0.001 / (10.0**0) * 48**0,
            0.0,
            1.0 * 0.001 / (10.0**2) * 48**2 * 2**1,
        ),
        polynomial_difficulties: tuple[float, ...] = (
            0.0,
            0.0,
            -10.0 * 0.001,
        ),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for **difficulty-based** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of an arbitrary combination of polynomial
        nonlinearities and (isotropic) linear derivatives. Uses a
        difficulty-based interface where the "intensity" of the dynamics reduces
        with increasing resolution. This is intended such that emulator learning
        problems on two resolutions are comparibly difficult.

        Different to `exponax.stepper.generic.NormalizedPolynomialStepper`, the
        dynamics are defined by difficulties. The difficulties are a different
        combination of normalized dynamics, `num_spatial_dims`, and
        `num_points`.

            γᵢ = αᵢ Nⁱ 2ⁱ⁻¹ d

        with `d` the number of spatial dimensions, `N` the number of points, and
        `αᵢ` the normalized coefficient.

        Since the polynomial nonlinearity does not contain any derivatives, we
        have that

        ```
            normalized_polynomial_scales = polynomial_difficulties
        ```

        The default settings correspond to the Fisher-KPP equation.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`.
        - `linear_difficulties`: The list of difficulties `γ_j` corresponding to
            the derivatives. The length of this tuple represents the highest
            occuring derivative. The default value corresponds to the Fisher-KPP
            equation.
        - `polynomial_difficulties`: The list of difficulties `δₖ` corresponding
            to the polynomial contributions. The length of this tuple represents
            the highest occuring polynomial. The default value corresponds to the
            Fisher-KPP equation.
        - `order`: The order of the Exponential Time Differencing Runge Kutta
            method. Must be one of {0, 1, 2, 3, 4}. The option `0` only solves
            the linear part of the equation. Use higher values for higher accuracy
            and stability. The default choice of `2` is a good compromise for
            single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep before
            evaluating the nonlinearity. The default 2/3 corresponds to Orszag's
            2/3 rule which is sufficient if the highest occuring polynomial is
            quadratic (i.e., there are at maximum three entries in the
            `polynomial_difficulties` tuple).
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta method.
        """
        self.linear_difficulties = linear_difficulties
        self.polynomial_difficulties = polynomial_difficulties

        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            linear_difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
        # For polynomial nonlinearities, we have difficulties == normalized scales
        normalized_polynomial_scales = polynomial_difficulties

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_linear_coefficients=normalized_coefficients,
            normalized_polynomial_coefficients=normalized_polynomial_scales,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )
