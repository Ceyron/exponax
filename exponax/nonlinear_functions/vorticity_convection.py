import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..spectral import build_laplace_operator, build_scaling_array, build_wavenumbers
from .base import BaseNonlinearFun


class VorticityConvection2d(BaseNonlinearFun):
    inv_laplacian: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        if num_spatial_dims != 2:
            raise ValueError(f"Expected num_spatial_dims = 2, got {num_spatial_dims}.")
        if num_channels != 1:
            raise ValueError(f"Expected num_channels = 1, got {num_channels}.")

        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

        laplacian = build_laplace_operator(derivative_operator, order=2)

        # Uses the UNCHANGED mean solution to the Poisson equation (hence, the
        # mean of the "right-hand side" will be the mean of the solution)
        self.inv_laplacian = jnp.where(laplacian == 0, 1.0, 1 / laplacian)

    def evaluate(
        self, u_hat: Complex[Array, "1 ... (N//2)+1"]
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        vorticity_hat = u_hat
        stream_function_hat = self.inv_laplacian * vorticity_hat

        u_hat = +self.derivative_operator[1:2] * stream_function_hat
        v_hat = -self.derivative_operator[0:1] * stream_function_hat
        del_vorticity_del_x_hat = self.derivative_operator[0:1] * vorticity_hat
        del_vorticity_del_y_hat = self.derivative_operator[1:2] * vorticity_hat

        u = jnp.fft.irfft2(
            u_hat * self.dealiasing_mask, s=(self.num_points, self.num_points)
        )
        v = jnp.fft.irfft2(
            v_hat * self.dealiasing_mask, s=(self.num_points, self.num_points)
        )
        del_vorticity_del_x = jnp.fft.irfft2(
            del_vorticity_del_x_hat * self.dealiasing_mask,
            s=(self.num_points, self.num_points),
        )
        del_vorticity_del_y = jnp.fft.irfft2(
            del_vorticity_del_y_hat * self.dealiasing_mask,
            s=(self.num_points, self.num_points),
        )

        convection = u * del_vorticity_del_x + v * del_vorticity_del_y

        convection_hat = jnp.fft.rfft2(convection)

        # Do we need another dealiasing mask here?
        # convection_hat = self.dealiasing_mask * convection_hat

        # Requires minus to move term to the rhs
        return -convection_hat


class VorticityConvection2dKolmogorov(VorticityConvection2d):
    injection: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

        wavenumbers = build_wavenumbers(num_spatial_dims, num_points)
        injection_mask = (wavenumbers[0] == 0) & (wavenumbers[1] == injection_mode)
        self.injection = jnp.where(
            injection_mask,
            injection_scale * build_scaling_array(num_spatial_dims, num_points),
            0.0,
        )

    def evaluate(
        self, u_hat: Complex[Array, "1 ... (N//2)+1"]
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        neg_convection_hat = super().evaluate(u_hat)
        return neg_convection_hat + self.injection
