import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Complex, Array, Float, Bool
from ..spectral import (
    space_indices,
    spatial_shape,
)

from .base import BaseNonlinearFun


class GradientNormNonlinearFun(BaseNonlinearFun):
    scale: float
    zero_mode_fix: bool

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        zero_mode_fix: bool = True,
        scale: float = 0.5,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.zero_mode_fix = zero_mode_fix
        self.scale = scale

    def zero_fix(
        self,
        f: Float[Array, "... N"],
    ):
        return f - jnp.mean(f)

    def evaluate(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_gradient_hat = self.derivative_operator[None, :] * u_hat[:, None]
        u_gradient_dealiased_hat = self.dealiasing_mask * u_gradient_hat
        u_gradient = jnp.fft.irfftn(
            u_gradient_dealiased_hat,
            s=spatial_shape(self.num_spatial_dims, self.num_points),
            axes=space_indices(self.num_spatial_dims),
        )

        # Reduces the axis introduced by the gradient
        u_gradient_norm_squared = jnp.sum(u_gradient**2, axis=1)

        if self.zero_mode_fix:
            # Maybe there is more efficient way
            u_gradient_norm_squared = jax.vmap(self.zero_fix)(u_gradient_norm_squared)

        u_gradient_norm_squared_hat = jnp.fft.rfftn(
            u_gradient_norm_squared, axes=space_indices(self.num_spatial_dims)
        )
        # if self.zero_mode_fix:
        #     # Fix the mean mode
        #     u_gradient_norm_squared_hat = u_gradient_norm_squared_hat.at[..., 0].set(
        #         u_hat[..., 0]
        #     )

        # Requires minus to move term to the rhs
        return -self.scale * u_gradient_norm_squared_hat
