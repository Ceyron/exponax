from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import ConvectionNonlinearFun


class Burgers(BaseStepper):
    diffusivity: float
    convection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.1,
        convection_scale: float = 1.0,
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
        `b₁ = 1`. In 1d, the state `u` has only one channel as such the state is
        represented by a tensor of shape `(1, num_points)`. For higher
        dimensions, the channels grow with the dimension, i.e. in 2d the state
        `u` is represented by a tensor of shape `(2, num_points, num_points)`.
        The equation in 2d reads using vector format for the two channels

        ```
            uₜ + b₁ 1/2 ∇ ⋅ (u ⊗ u) = ν Δu
        ```

        with `∇ ⋅` the divergence operator and `Δ` the Laplacian.

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
        """
        # """
        # Convection is always scaled by 0.5, use `convection_scale` to multiply
        # an additional factor.
        # """
        self.diffusivity = diffusivity
        self.convection_scale = convection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=num_spatial_dims,  # Number of channels grows with dimension
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
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.convection_scale,
        )
