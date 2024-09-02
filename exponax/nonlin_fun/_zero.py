import jax.numpy as jnp
from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class ZeroNonlinearFun(BaseNonlinearFun):
    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
    ):
        """
        A zero nonlinear function to be used for linear problems. It's nonlinear
        function is just

        ```
            𝒩(u) = 0
        ```

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same.
        """
        super().__init__(
            num_spatial_dims,
            num_points,
        )

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return jnp.zeros_like(u_hat)
