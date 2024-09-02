from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ..._spectral import build_laplace_operator
from ...nonlin_fun import BaseNonlinearFun


class CahnHilliardNonlinearFun(BaseNonlinearFun):
    scale: float
    laplace_operator: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        scale: float,
        dealiasing_fraction: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.laplace_operator = build_laplace_operator(derivative_operator)
        self.scale = scale

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u = self.ifft(self.dealias(u_hat))
        u_power = u[0] ** 3
        u_power_hat = self.fft(u_power)
        u_power_laplace_hat = self.laplace_operator * u_power_hat
        return u_power_laplace_hat * self.scale


class CahnHilliard(BaseStepper):
    diffusivity: float
    gamma: float
    first_order_coefficient: float
    third_order_coefficient: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 1e-2,
        gamma: float = 1e-3,
        first_order_coefficient: float = -1.0,
        third_order_coefficient: float = 1.0,
        order: int = 2,
        # Needs lower value due to cubic nonlinearity
        dealiasing_fraction: float = 1 / 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Cahn-Hilliard
        reaction-diffusion equation on periodic boundary conditions. This model
        is related to the Allen-Cahn equation. In 1d, it reads

        ```
            uₜ = ν ∂ₓₓ (c₃ u³ + c₁ u − γ uₓₓ)
        ```

        with `ν` the diffusivity, `c₁` the first order coefficient, `c₃` the
        third order coefficient, and `γ` the gamma parameter. The state always
        only has one channel. In higher dimensions, the equation reads

        ```
            uₜ = ν Δ (c₃ u³ + c₁ u - γ Δu)
        ```

        Since the Laplace operator is self-multiplied, there will be spatial
        mixing.

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
        - `gamma`: The gamma parameter `γ`.
        - `first_order_coefficient`: The first order coefficient `c₁`.
        - `third_order_coefficient`: The third order coefficient `c₃`.
        - `dealiasing_fraction`: The fraction of the highest wavenumbers to
            dealias. Default is `1/2` because the default polynomial has a
            highest degree of 3.
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

        **Notes:**

        - See
            https://github.com/chebfun/chebfun/blob/db207bc9f48278ca4def15bf90591bfa44d0801d/spin.m#L89
            for an example IC of the Cahn-Hilliard in 1d.
        """
        self.diffusivity = diffusivity
        self.gamma = gamma
        self.first_order_coefficient = first_order_coefficient
        self.third_order_coefficient = third_order_coefficient
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
        laplace = build_laplace_operator(derivative_operator, order=2)
        linear_operator = (
            self.diffusivity
            * laplace
            * (self.first_order_coefficient - self.gamma * laplace)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> CahnHilliardNonlinearFun:
        return CahnHilliardNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.diffusivity * self.third_order_coefficient,
        )
