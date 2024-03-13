import jax.numpy as jnp
from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class ConvectionNonlinearFun(BaseNonlinearFun):
    derivative_operator: Complex[Array, "D ... (N//2)+1"]
    scale: float
    single_channel: bool

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float = 2 / 3,
        scale: float = 1.0,
        single_channel: bool = False,
    ):
        """
        Uses by default a scaling of 0.5 to take into account the conservative evaluation
        """
        self.derivative_operator = derivative_operator
        self.scale = scale
        self.single_channel = single_channel
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

    def _multi_channel_eval(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        num_channels = u_hat.shape[0]
        if num_channels != self.num_spatial_dims:
            raise ValueError(
                "Number of channels in u_hat should match number of spatial dimensions"
            )
        u_hat_dealiased = self.dealias(u_hat)
        u = self.ifft(u_hat_dealiased)
        u_outer_product = u[:, None] * u[None, :]
        u_outer_product_hat = self.fft(u_outer_product)
        convection = 0.5 * jnp.sum(
            self.derivative_operator[None, :] * u_outer_product_hat,
            axis=1,
        )
        # Requires minus to move term to the rhs
        return -self.scale * convection

    def _single_channel_eval(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_hat_dealiased = self.dealias(u_hat)
        u = self.ifft(u_hat_dealiased)
        u_square = u**2
        u_square_hat = self.fft(u_square)
        sum_of_derivatives_operator = jnp.sum(
            self.derivative_operator, axis=0, keepdims=True
        )
        convection = 0.5 * sum_of_derivatives_operator * u_square_hat
        # Requires minus to bring convection to the right-hand side
        return -self.scale * convection

    def __call__(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        if self.single_channel:
            return self._single_channel_eval(u_hat)
        else:
            return self._multi_channel_eval(u_hat)
