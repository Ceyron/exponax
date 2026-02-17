import jax.numpy as jnp
from jaxtyping import Array, Complex, Inexact

from .._spectral import build_scaling_array, build_wavenumbers
from ._base import BaseNonlinearFun
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


class ProjectedConvection3dKolmogorov(ProjectedConvection3d):
    injection: Complex[Array, "3 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        derivative_operator: Complex[Array, " 3 ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

        wavenumbers = build_wavenumbers(num_spatial_dims, num_points)
        injection_mask = (
            (wavenumbers[0] == 0)
            & (wavenumbers[1] == injection_mode)
            & (wavenumbers[2] == 0)
        )
        # In 3D, we work with velocity directly (not vorticity), so the
        # forcing is f = γ sin(k x₁) ê₀. Only the first velocity channel is
        # forced, and no extra -k factor is needed (unlike the 2D vorticity
        # formulation).
        injection_single = jnp.where(
            injection_mask,
            injection_scale
            * build_scaling_array(num_spatial_dims, num_points, mode="coef_extraction"),
            0.0,
        )
        # Shape (3, N, N, N//2+1): forcing only in the first velocity channel
        zeros = jnp.zeros_like(injection_single)
        self.injection = jnp.concatenate([injection_single, zeros, zeros], axis=0)

    def __call__(
        self, u_hat: Complex[Array, "3 ... (N//2)+1"]
    ) -> Complex[Array, "3 ... (N//2)+1"]:
        neg_convection_hat = super().__call__(u_hat)
        return neg_convection_hat + self.injection
