import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .._spectral import spatial_shape
from ._base_ic import BaseRandomICGenerator


class WhiteNoise(BaseRandomICGenerator):
    """
    Raw white noise generator that samples i.i.d. normal noise in physical
    (state) space.

    This is a building block for other IC generators. It does **not** apply any
    normalization (zero-mean, std-one, max-one) -- that is left to the outer
    generator that composes with this class.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `std`: Standard deviation of the noise. Defaults to `1.0`.
    """

    num_spatial_dims: int
    std: float = 1.0

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        noise = self.std * jr.normal(
            key, shape=(1,) + spatial_shape(self.num_spatial_dims, num_points)
        )
        return noise
