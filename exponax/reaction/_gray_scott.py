import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import BaseNonlinearFun


class GrayScottNonlinearFun(BaseNonlinearFun):
    b: float
    d: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: float,
        b: float,
        d: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.b = b
        self.d = d

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        num_channels = u_hat.shape[0]
        if num_channels != 2:
            raise ValueError("num_channels must be 2")
        u = self.ifft(self.dealias(u_hat))
        u_power = jnp.stack(
            [
                self.b * (1 - u[0]) - u[0] * u[1] ** 2,
                -self.d * u[1] + u[0] * u[1] ** 2,
            ]
        )
        u_power_hat = self.fft(u_power)
        return u_power_hat


class GrayScott(BaseStepper):
    epsilon_1: float
    epsilon_2: float
    b: float
    d: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        epsilon_1: float = 0.00002,
        epsilon_2: float = 0.00001,
        b: float = 0.04,
        d: float = 0.1,
        order: int = 2,
        dealiasing_fraction: float = 1
        / 2,  # Needs lower value due to cubic nonlinearity
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        See also this papers:
        https://www.ljll.fr/hecht/ftp/ff++/2015-cimpa-IIT/edp-tuto/Pearson.pdf

        There the two parameters are called F and k, named feed rate and kill
        rate. The arguments to this equation are such that b=F and d=F+k. The
        paper used the domain extent of 2.5. The epsilon values (=the two
        diffusivities) are the same.
        """
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.b = b
        self.d = d
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=2,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "2 ... (N//2)+1"]:
        laplace = build_laplace_operator(derivative_operator, order=2)
        linear_operator = jnp.concatenate(
            [
                self.epsilon_1 * laplace,
                self.epsilon_2 * laplace,
            ]
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> GrayScottNonlinearFun:
        return GrayScottNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            b=self.b,
            d=self.d,
            dealiasing_fraction=self.dealiasing_fraction,
        )
