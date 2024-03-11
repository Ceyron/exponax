import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .._spectral import spatial_shape
from ..stepper import Diffusion
from ._base_ic import BaseRandomICGenerator


class DiffusedNoise(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    intensity: float
    zero_mean: bool

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        domain_extent: float = 1.0,
        intensity=0.001,
        zero_mean: bool = False,
    ):
        """
        Randomly generated initial condition consisting of a diffused noise field.

        Arguments are drawn from uniform distributions.

        **Arguments**:
            - `D`: The dimension of the domain.
            - `L`: The length of the domain.
            - `N`: The number of grid points in each dimension.
            - `intensity`: The diffusivity.
            - `zero_mean`: Whether to subtract the mean.
        """
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.intensity = intensity
        self.zero_mean = zero_mean

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        noise_shape = (1,) + spatial_shape(self.num_spatial_dims, num_points)
        noise = jr.normal(key, shape=noise_shape)

        diffusion_stepper = Diffusion(
            self.num_spatial_dims,
            self.domain_extent,
            num_points,
            1.0,
            diffusivity=self.intensity,
        )
        ic = diffusion_stepper(noise)

        if self.zero_mean:
            ic = ic - jnp.mean(ic)

        return ic
