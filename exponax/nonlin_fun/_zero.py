import jax.numpy as jnp
from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class ZeroNonlinearFun(BaseNonlinearFun):
    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
        )

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return jnp.zeros_like(u_hat)
