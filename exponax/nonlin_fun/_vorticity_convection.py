import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._spectral import build_laplace_operator, build_scaling_array, build_wavenumbers
from ._base import BaseNonlinearFun


class VorticityConvection2d(BaseNonlinearFun):
    convection_scale: float
    derivative_operator: Complex[Array, "D ... (N//2)+1"]
    inv_laplacian: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        convection_scale: float = 1.0,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        if num_spatial_dims != 2:
            raise ValueError(f"Expected num_spatial_dims = 2, got {num_spatial_dims}.")

        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

        self.convection_scale = convection_scale
        self.derivative_operator = derivative_operator

        laplacian = build_laplace_operator(derivative_operator, order=2)

        # Uses the UNCHANGED mean solution to the Poisson equation (hence, the
        # mean of the "right-hand side" will be the mean of the solution)
        self.inv_laplacian = jnp.where(laplacian == 0, 1.0, 1 / laplacian)

    def __call__(
        self, u_hat: Complex[Array, "1 ... (N//2)+1"]
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        vorticity_hat = u_hat
        stream_function_hat = self.inv_laplacian * vorticity_hat

        u_hat = +self.derivative_operator[1:2] * stream_function_hat
        v_hat = -self.derivative_operator[0:1] * stream_function_hat
        del_vorticity_del_x_hat = self.derivative_operator[0:1] * vorticity_hat
        del_vorticity_del_y_hat = self.derivative_operator[1:2] * vorticity_hat

        u = self.ifft(self.dealias(u_hat))
        v = self.ifft(self.dealias(v_hat))
        del_vorticity_del_x = self.ifft(self.dealias(del_vorticity_del_x_hat))
        del_vorticity_del_y = self.ifft(self.dealias(del_vorticity_del_y_hat))

        convection = u * del_vorticity_del_x + v * del_vorticity_del_y

        convection_hat = self.fft(convection)

        # Need minus to bring term to the right-hand side
        return -self.convection_scale * convection_hat


class VorticityConvection2dKolmogorov(VorticityConvection2d):
    injection: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        convection_scale: float = 1.0,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            convection_scale=convection_scale,
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

    def __call__(
        self, u_hat: Complex[Array, "1 ... (N//2)+1"]
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        neg_convection_hat = super().__call__(u_hat)
        return neg_convection_hat + self.injection
