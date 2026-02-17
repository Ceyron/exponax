import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from .._spectral import build_scaled_wavenumbers
from ..nonlin_fun import ZeroNonlinearFun


class Wave(BaseStepper):
    speed_of_sound: float
    wavenumber_norm: Float[Array, " 1 ... N "]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        speed_of_sound: float = 1.0,
    ):
        self.speed_of_sound = speed_of_sound
        self.wavenumber_norm = jnp.linalg.norm(
            build_scaled_wavenumbers(
                num_spatial_dims=num_spatial_dims,
                domain_extent=domain_extent,
                num_points=num_points,
            ),
            axis=0,
            keepdims=True,
        )
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=2,
            order=0,
        )

    def _forward_transform(
        self, u_hat: Complex[Array, " 2 ... N "]
    ) -> Complex[Array, " 2 ... N "]:
        h_hat, v_hat = u_hat[0:1], u_hat[1:2]
        # Pre-scale height so units match velocity
        # w = i * c * |k| * h
        k_guard = jnp.where(self.wavenumber_norm == 0, 1.0, self.wavenumber_norm)
        w_hat = 1j * self.speed_of_sound * k_guard * h_hat

        # Standard Orthonormal (Unitary) rotation
        pos = (1 / jnp.sqrt(2)) * (w_hat + v_hat)
        neg = (1 / jnp.sqrt(2)) * (w_hat - v_hat)
        return jnp.concatenate([pos, neg], axis=0)

    def _inverse_transform(
        self, waves_hat: Complex[Array, " 2 ... N "]
    ) -> Complex[Array, " 2 ... N "]:
        pos, neg = waves_hat[0:1], waves_hat[1:2]
        # Inverse rotation
        w_hat = (1 / jnp.sqrt(2)) * (pos + neg)
        v_hat = (1 / jnp.sqrt(2)) * (pos - neg)

        # Back to physical height
        k_guard = jnp.where(self.wavenumber_norm == 0, 1.0, self.wavenumber_norm)
        h_hat = w_hat / (1j * self.speed_of_sound * k_guard)
        return jnp.concatenate([h_hat, v_hat], axis=0)

    def _build_linear_operator(
        self, derivative_operator: Complex[Array, " D ... N "]
    ) -> Complex[Array, " 2 ... N "]:
        val = 1j * self.speed_of_sound * self.wavenumber_norm
        return jnp.concatenate(
            (
                val,
                -val,
            ),
            axis=0,
        )

    def _build_nonlinear_fun(
        self, derivative_operator: Complex[Array, " D ... N "]
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(self.num_spatial_dims, self.num_points)

    # Overvride step_fourier to include transforms
    def step_fourier(
        self, u_hat: Complex[Array, " 2 ... N "]
    ) -> Complex[Array, " 2 ... N "]:
        waves_hat = self._forward_transform(u_hat)
        waves_hat_next = super().step_fourier(waves_hat)
        u_hat_next = self._inverse_transform(waves_hat_next)

        # The k=0 (mean/DC) mode cannot be diagonalized because the two
        # eigenvalues collapse to zero and the system matrix becomes
        # [[0, 1], [0, 0]]. The diagonalization leaves this mode unchanged,
        # but the exact solution is h_mean(t+dt) = h_mean(t) + dt * v_mean(t).
        # Apply this linear drift explicitly.
        h_dc_idx = (0,) + (0,) * self.num_spatial_dims
        v_dc_idx = (1,) + (0,) * self.num_spatial_dims
        u_hat_next = u_hat_next.at[h_dc_idx].add(self.dt * u_hat[v_dc_idx])

        return u_hat_next
