import jax.numpy as jnp
from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class ConvectionNonlinearFun(BaseNonlinearFun):
    derivative_operator: Complex[Array, "D ... (N//2)+1"]
    scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        scale: float = 1.0,
    ):
        """
        Uses by default a scaling of 0.5 to take into account the conservative evaluation
        """
        self.derivative_operator = derivative_operator
        self.scale = scale
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

    def __call__(
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

    # def evaluate(
    #     self,
    #     u_hat: Complex[Array, "C ... (N//2)+1"],
    # ) -> Complex[Array, "C ... (N//2)+1"]:
    #     u_hat_dealiased = self.dealiasing_mask * u_hat
    #     u = jnp.fft.irfftn(
    #         u_hat_dealiased,
    #         s=spatial_shape(self.num_spatial_dims, self.num_points),
    #         axes=space_indices(self.num_spatial_dims),
    #     )
    #     u_outer_product = u[:, None] * u[None, :]

    #     u_outer_product_hat = jnp.fft.rfftn(
    #         u_outer_product, axes=space_indices(self.num_spatial_dims)
    #     )
    #     u_divergence_on_outer_product_hat = jnp.sum(
    #         self.derivative_operator[None, :] * u_outer_product_hat,
    #         axis=1,
    #     )
    #     # Requires minus to move term to the rhs
    #     return -self.scale * 0.5 * u_divergence_on_outer_product_hat
