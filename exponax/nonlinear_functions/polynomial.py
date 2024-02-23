import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Complex, Array, Float, Bool
from ..spectral import (
    space_indices,
    spatial_shape,
)

from .base import BaseNonlinearFun


class PolynomialNonlinearFun(BaseNonlinearFun):
    """
    Channel-separate evaluation; and no mixed terms.
    """

    coefficients: list[float]  # Starting from order 0

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        coefficients: list[float],
    ):
        """
        Coefficient list starts from order 0.
        """
        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.coefficients = coefficients

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
        u_power = 1.0
        u_nonlin = 0.0
        for coeff in self.coefficients:
            u_nonlin += coeff * u_power
            u_power = u_power * u

        u_nonlin_hat = jnp.fft.rfftn(
            u_nonlin, axes=space_indices(self.num_spatial_dims)
        )
        return u_nonlin_hat
