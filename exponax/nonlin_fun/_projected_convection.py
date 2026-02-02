import jax.numpy as jnp
from jaxtyping import Array, Complex, Inexact

from exponax.nonlin_fun import BaseNonlinearFun

from ._leray import Leray


def _cross_product_3d(
    a: Inexact[Array, " 3 N N (N//2+1) "],
    b: Inexact[Array, " 3 N N (N//2+1) "],
) -> Inexact[Array, " 3 N N (N//2+1) "]:
    c1 = a[1] * b[2] - a[2] * b[1]
    c2 = a[2] * b[0] - a[0] * b[2]
    c3 = a[0] * b[1] - a[1] * b[0]
    return jnp.stack([c1, c2, c3], axis=0)


class ProjectedConvection3d(BaseNonlinearFun):
    derivative_operator: Complex[Array, " 3 N N (N//2+1) "]
    leray_projection: Leray

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, " 3 N N (N//2+1) "],
        dealiasing_fraction: float = 2 / 3,
    ):
        """
        Based on

        https://arxiv.org/pdf/1602.03638
        """
        if num_spatial_dims != 3:
            raise ValueError(
                "ProjectedConvection3d only supports 3 spatial dimensions."
            )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

        self.derivative_operator = derivative_operator

        self.leray_projection = Leray(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            derivative_operator=derivative_operator,
        )

    def __call__(
        self,
        u_hat: Complex[Array, " 3 N N (N//2+1) "],
    ) -> Complex[Array, " 3 N N (N//2+1) "]:
        velocity_hat = u_hat
        curl_hat = _cross_product_3d(
            self.derivative_operator,
            velocity_hat,
        )

        curl = self.ifft(self.dealias(curl_hat))
        velocity = self.ifft(self.dealias(velocity_hat))

        convection = _cross_product_3d(
            velocity,
            curl,
        )

        convection_hat = self.fft(convection)

        convection_projected_hat = self.leray_projection(convection_hat)

        return convection_projected_hat
