from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import BaseNonlinearFun


class CahnHilliardNonlinearFun(BaseNonlinearFun):
    laplace_operator: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.laplace_operator = build_laplace_operator(derivative_operator)

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u = self.ifft(self.dealias(u_hat))
        u_power = u[0] ** 3
        u_power_hat = self.fft(u_power)
        u_power_laplace_hat = self.laplace_operator * u_power_hat
        return u_power_laplace_hat


class CahnHilliard(BaseStepper):
    hyper_diffusivity: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        hyper_diffusivity: float = 0.2,
        order: int = 2,
        dealiasing_fraction: float = 1
        / 2,  # Needs lower value due to cubic nonlinearity
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.hyper_diffusivity = hyper_diffusivity
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
        bi_laplace = build_laplace_operator(derivative_operator, order=4)
        linear_operator = -self.hyper_diffusivity * bi_laplace - laplace
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
        )
