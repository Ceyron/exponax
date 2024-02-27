import jax.numpy as jnp
from jaxtyping import Array, Complex

from .base import BaseNonlinearFun


class ZeroNonlinearFun(BaseNonlinearFun):
    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float = 1.0,
    ):
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
        return jnp.zeros_like(u_hat)
