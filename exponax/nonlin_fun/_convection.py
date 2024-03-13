import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._spectral import space_indices, spatial_shape
from ._base import BaseNonlinearFun


class ConvectionNonlinearFun(BaseNonlinearFun):
    scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        scale: float = 1.0,
    ):
        """
        Uses by default a scaling of 0.5 to take into account the conservative evaluation
        """
        self.scale = scale
        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

    def evaluate(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_hat_dealiased = self.dealiasing_mask * u_hat
        u = jnp.fft.irfftn(
            u_hat_dealiased,
            s=spatial_shape(self.num_spatial_dims, self.num_points),
            axes=space_indices(self.num_spatial_dims),
        )
        u_outer_product = u[:, None] * u[None, :]

        u_outer_product_hat = jnp.fft.rfftn(
            u_outer_product, axes=space_indices(self.num_spatial_dims)
        )
        u_divergence_on_outer_product_hat = jnp.sum(
            self.derivative_operator[None, :] * u_outer_product_hat,
            axis=1,
        )
        # Requires minus to move term to the rhs
        return -self.scale * 0.5 * u_divergence_on_outer_product_hat
