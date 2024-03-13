import jax.numpy as jnp
from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class SingleChannelConvectionNonlinearFun(BaseNonlinearFun):
    sum_of_derivatives_operator: Complex[Array, "1 ... (N//2)+1"]
    scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float = 2 / 3,
        scale: float = 1.0,
    ):
        """
        Use additional default scaling of 0.5 to account for conservative eval.

        In contrast to the classical convection function, this one does not grow
        in channels as the number of spatial dimensions grow.
        """
        self.scale = scale
        self.sum_of_derivatives_operator = jnp.sum(
            derivative_operator, axis=0, keepdims=True
        )
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

    def __call__(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_hat_dealiased = self.dealias(u_hat)
        u = self.ifft(u_hat_dealiased)
        u_square = u**2
        u_square_hat = self.fft(u_square)
        single_channel_convection = (
            0.5 * self.sum_of_derivatives_operator * u_square_hat
        )
        # Requires minus to bring convection to the right-hand side
        return -self.scale * single_channel_convection
