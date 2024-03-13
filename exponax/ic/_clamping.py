import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._base_ic import BaseRandomICGenerator


class ClampingICGenerator(BaseRandomICGenerator):
    ic_gen: BaseRandomICGenerator
    limits: tuple[float, float]

    def __init__(
        self, ic_gen: BaseRandomICGenerator, limits: tuple[float, float] = (0, 1)
    ):
        self.ic_gen = ic_gen
        self.limits = limits
        self.num_spatial_dims = ic_gen.num_spatial_dims

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        ic = self.ic_gen(num_points=num_points, key=key)
        ic_above_zero = ic - jnp.min(ic)
        ic_clamped_to_unit_limits = ic_above_zero / jnp.max(ic_above_zero)
        range = self.limits[1] - self.limits[0]
        return ic_clamped_to_unit_limits * range + self.limits[0]
