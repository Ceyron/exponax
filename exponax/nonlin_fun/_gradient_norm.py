import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._base import BaseNonlinearFun


class GradientNormNonlinearFun(BaseNonlinearFun):
    scale: float
    zero_mode_fix: bool
    derivative_operator: Complex[Array, "D ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        zero_mode_fix: bool = True,
        scale: float = 1.0,
    ):
        """
        Uses by default a scaling of 0.5 to take into account the conservative evaluation
        """
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.derivative_operator = derivative_operator
        self.zero_mode_fix = zero_mode_fix
        self.scale = scale

    def zero_fix(
        self,
        f: Float[Array, "... N"],
    ):
        return f - jnp.mean(f)

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_gradient_hat = self.derivative_operator[None, :] * u_hat[:, None]
        u_gradient = self.ifft(self.dealias(u_gradient_hat))

        # Reduces the axis introduced by the gradient
        u_gradient_norm_squared = jnp.sum(u_gradient**2, axis=1)

        if self.zero_mode_fix:
            # Maybe there is more efficient way
            u_gradient_norm_squared = jax.vmap(self.zero_fix)(u_gradient_norm_squared)

        u_gradient_norm_squared_hat = 0.5 * self.fft(u_gradient_norm_squared)
        # if self.zero_mode_fix:
        #     # Fix the mean mode
        #     u_gradient_norm_squared_hat = u_gradient_norm_squared_hat.at[..., 0].set(
        #         u_hat[..., 0]
        #     )

        # Requires minus to move term to the rhs
        return -self.scale * u_gradient_norm_squared_hat
