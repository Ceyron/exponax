import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import BaseNonlinearFun


class GrayScottNonlinearFun(BaseNonlinearFun):
    feed_rate: float
    kill_rate: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: float,
        feed_rate: float,
        kill_rate: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate

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
                self.feed_rate * (1 - u[0]) - u[0] * u[1] ** 2,
                -(self.feed_rate + self.kill_rate) * u[1] + u[0] * u[1] ** 2,
            ]
        )
        u_power_hat = self.fft(u_power)
        return u_power_hat


class GrayScott(BaseStepper):
    diffusivity_1: float
    diffusivity_2: float
    feed_rate: float
    kill_rate: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity_1: float = 2e-5,
        diffusivity_2: float = 1e-5,
        feed_rate: float = 0.04,
        kill_rate: float = 0.06,
        order: int = 2,
        # Needs lower value due to cubic nonlinearity
        dealiasing_fraction: float = 1 / 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        See also this papers:
        https://www.ljll.fr/hecht/ftp/ff++/2015-cimpa-IIT/edp-tuto/Pearson.pdf
        """
        self.diffusivity_1 = diffusivity_1
        self.diffusivity_2 = diffusivity_2
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate
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
                self.diffusivity_1 * laplace,
                self.diffusivity_2 * laplace,
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
            feed_rate=self.feed_rate,
            kill_rate=self.kill_rate,
            dealiasing_fraction=self.dealiasing_fraction,
        )
