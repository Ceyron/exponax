import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ...nonlin_fun import ConvectionNonlinearFun
from ._utils import (
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_convection_scale_from_difficulty,
)


class GeneralConvectionStepper(BaseStepper):
    linear_coefficients: tuple[float, ...]
    convection_scale: float
    dealiasing_fraction: float
    single_channel: bool
    conservative: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        linear_coefficients: tuple[float, ...] = (0.0, 0.0, 0.01),
        convection_scale: float = 1.0,
        single_channel: bool = False,
        conservative: bool = False,
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

        Depending on the collection of linear coefficients a range of dynamics
        can be represented, for example:
            - Burgers equation with `a = (0, 0, 0.01)` with `len(a) = 3`
            - KdV equation with `a = (0, 0, 0, 0.01)` with `len(a) = 4`

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`.
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
            0.0, 0.01)` corresponds to the Burgers equation (because of the
            diffusion)
        - `convection_scale` (keyword-only): The scale `b₁` of the
            convection term. Default is `1.0`.
        - `single_channel`: Whether to use the single channel mode in higher
            dimensions. In this case the the convection is `b₁ (∇ ⋅ 1)(u²)`. In
            this case, the state always has a single channel, no matter the
            spatial dimension. Default: False.
        - `conservative`: Whether to use the conservative form of the convection
            term. Default: False.
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
        self.convection_scale = convection_scale
        self.single_channel = single_channel
        self.dealiasing_fraction = dealiasing_fraction
        self.conservative = conservative

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
            for i, c in enumerate(self.linear_coefficients)
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


class NormalizedConvectionStepper(GeneralConvectionStepper):
    normalized_linear_coefficients: tuple[float, ...]
    normalized_convection_scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_linear_coefficients: tuple[float, ...] = (0.0, 0.0, 0.01 * 0.1),
        normalized_convection_scale: float = 1.0 * 0.1,
        single_channel: bool = False,
        conservative: bool = False,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Time stepper for the **normalized** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of a convection nonlinearity and an
        arbitrary combination of (isotropic) linear derivatives. Uses a
        normalized interface, i.e., the domain is scaled to `Ω = (0, 1)ᵈ` and
        time step size is `Δt = 1.0`.

        See `exponax.stepper.generic.GeneralConvectionStepper` for more details
        on the functional form of the PDE.

        In the default configuration, the number of channel grows with the
        number of spatial dimensions. Setting the flag `single_channel=True`
        activates a single-channel hack.

        Under the default settings, it behaves like the Burgers equation under
        the following settings

        ```python

        exponax.stepper.Burgers(
            D=D, L=1, N=N, dt=0.1, diffusivity=0.01,
        )
        ```

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`.
        - `normalized_linear_coefficients`: The list of coefficients
            `α_j` corresponding to the derivatives. The length of this tuple
            represents the highest occuring derivative. The default value `(0.0,
            0.0, 0.01)` corresponds to the Burgers equation (because of the
            diffusion contribution). Note that these coefficients are normalized
            on the unit domain and unit time step size.
        - `normalized_convection_scale`: The scale `β` of the convection term.
            Default is `1.0`.
        - `single_channel`: Whether to use the single channel mode in higher
            dimensions. In this case the the convection is `β (∇ ⋅ 1)(u²)`. In
            this case, the state always has a single channel, no matter the
            spatial dimension. Default: False.
        - `conservative`: Whether to use the conservative form of the convection
            term. Default: False.
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
        self.normalized_convection_scale = normalized_convection_scale
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
            linear_coefficients=normalized_linear_coefficients,
            convection_scale=normalized_convection_scale,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
            single_channel=single_channel,
            conservative=conservative,
        )


class DifficultyConvectionStepper(NormalizedConvectionStepper):
    linear_difficulties: tuple[float, ...]
    convection_difficulty: float

    def __init__(
        self,
        num_spatial_dims: int = 1,
        num_points: int = 48,
        *,
        linear_difficulties: tuple[float, ...] = (0.0, 0.0, 4.5),
        convection_difficulty: float = 5.0,
        single_channel: bool = False,
        conservative: bool = False,
        maximum_absolute: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the **difficulty-based** d-dimensional (`d ∈ {1, 2, 3}`)
        semi-linear PDEs consisting of a convection nonlinearity and an
        arbitrary combination of (isotropic) linear derivatives. Uses a
        difficulty-based interface where the "intensity" of the dynamics reduces
        with increasing resolution. This is intended such that emulator learning
        problems on two resolutions are comparibly difficult.

        Different to `exponax.stepper.generic.NormalizedConvectionStepper`, the
        dynamics are defined by difficulties. The difficulties are a different
        combination of normalized dynamics, `num_spatial_dims`, and
        `num_points`.

            γᵢ = αᵢ Nⁱ 2ⁱ⁻¹ d

        with `d` the number of spatial dimensions, `N` the number of points, and
        `αᵢ` the normalized coefficient.

        The difficulty of the nonlinear convection scale is defined by

            δ₁ = β₁ * M * N * D

        with `M` the maximum absolute value of the input state (typically `1.0`
        if one uses the `exponax.ic` random generators with the `max_one=True`
        argument).

        This interface is more natural than the normalized interface because the
        difficulties for all orders (given by `i`) are around 1.0. Additionally,
        they relate to stability condition of explicit Finite Difference schemes
        for the particular equations. For example, for advection (`i=1`), the
        absolute of the difficulty is the Courant-Friedrichs-Lewy (CFL) number.

        Under the default settings, this timestepper represents the Burgers
        equation.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same. Hence, the total number of degrees of freedom
            is `Nᵈ`.
        - `linear_difficulties`: The list of difficulties `γᵢ` corresponding to
            the derivatives. The length of this tuple represents the highest
            occuring derivative. The default value `(0.0, 0.0, 4.5)` corresponds
            to the Burgers equation. Note that these coefficients are normalized
            on the unit domain and unit time step size.
        - `convection_difficulty`: The difficulty `δ` of the convection term.
            Default is `5.0`.
        - `single_channel`: Whether to use the single channel mode in higher
            dimensions. In this case the the convection is `δ (∇ ⋅ 1)(u²)`. In
            this case, the state always has a single channel, no matter the
            spatial dimension. Default: False.
        - `conservative`: Whether to use the conservative form of the convection
        - `maximum_absolute`: The maximum absolute value of the state. This is
            used to extract the normalized dynamics from the convection
            difficulty.
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
        self.convection_difficulty = convection_difficulty
        normalized_coefficients = extract_normalized_coefficients_from_difficulty(
            linear_difficulties,
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )
        normalized_convection_scale = (
            extract_normalized_convection_scale_from_difficulty(
                convection_difficulty,
                num_spatial_dims=num_spatial_dims,
                num_points=num_points,
                maximum_absolute=maximum_absolute,
            )
        )
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            normalized_linear_coefficients=normalized_coefficients,
            normalized_convection_scale=normalized_convection_scale,
            single_channel=single_channel,
            order=order,
            dealiasing_fraction=dealiasing_fraction,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
            conservative=conservative,
        )
