from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import ConvectionNonlinearFun


class Burgers(BaseStepper):
    diffusivity: float
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
        diffusivity: float = 0.1,
        convection_scale: float = 1.0,
        single_channel: bool = False,
        conservative: bool = False,
        order=2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Burgers equation on
        periodic boundary conditions.

        In 1d, the Burgers equation is given by

        ```
            uₜ + b₁ 1/2 (u²)ₓ = ν uₓₓ
        ```

        with `b₁` the convection coefficient and `ν` the diffusivity. Oftentimes
        `b₁ = 1`. In 1d, the state `u` has only one channel. As such the
        discretized state is represented by a tensor of shape `(1, num_points)`.
        For higher dimensions, the channels grow with the dimension, i.e. in 2d
        the state `u` is represented by a tensor of shape `(2, num_points,
        num_points)`. The equation in 2d reads (using vector format for the two
        channels)

        ```
            uₜ + b₁ 1/2 ∇ ⋅ (u ⊗ u) = ν Δu
        ```

        with `∇ ⋅` the divergence operator and `Δ` the Laplacian.

        The expected temporal behavior is that the initial condition becomes
        "sharper"; in 1d positive values move to the right and negative values
        to the left. Smooth shocks develop that propagate at speed depending on
        the height difference. Ultimately, the solution decays to a constant
        state.

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
        - `diffusivity`: The diffusivity `ν` in the Burgers equation.
            Default: 0.1.
        - `convection_scale`: The scaling factor for the convection term.
            Note that the scaling by 1/2 is always performed. Default: 1.0.
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

        **Notes:**

        - If the `diffusivity` is set too low, spurious oscillations may
            occur because the solution becomes "too discontinous". Such
            simulations are not possible with Fourier pseudospectral methods.
            Sometimes increasing the number of points `N` can help.

        **Good Values:**

        - Next to the defaults of `diffusivity=0.1` and
            `convection_scale=1.0`, the following values are good starting
            points:
            - `num_points=100` for 1d.
            - `domain_extent=1`
            - `dt=0.1`
            - A bandlimited initial condition with maximum absolute value of
                ~1.0
        """
        self.diffusivity = diffusivity
        self.convection_scale = convection_scale
        self.single_channel = single_channel
        self.conservative = conservative
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
        # The linear operator is the same for all D channels
        return self.diffusivity * build_laplace_operator(derivative_operator)

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
