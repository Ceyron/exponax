import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .._spectral import (
    build_scaled_wavenumbers,
    space_indices,
    spatial_shape,
    wavenumber_shape,
)
from ._base_ic import BaseRandomICGenerator


class GaussianRandomField(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    powerlaw_exponent: float
    zero_mean: bool
    max_one: bool

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        domain_extent: float = 1.0,
        powerlaw_exponent: float = 3.0,
        zero_mean: bool = True,
        max_one: bool = False,
    ):
        """
        Randomly generated initial condition consisting of a Gaussian random field.
        """
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.powerlaw_exponent = powerlaw_exponent
        self.zero_mean = zero_mean
        self.max_one = max_one

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        wavenumber_grid = build_scaled_wavenumbers(
            self.num_spatial_dims, self.domain_extent, num_points
        )
        wavenumer_norm_grid = jnp.linalg.norm(wavenumber_grid, axis=0, keepdims=True)
        amplitude = jnp.power(wavenumer_norm_grid, -self.powerlaw_exponent / 2.0)
        amplitude = (
            amplitude.flatten().at[0].set(0.0).reshape(wavenumer_norm_grid.shape)
        )

        real_key, imag_key = jr.split(key, 2)
        noise = jr.normal(
            real_key,
            shape=(1,) + wavenumber_shape(self.num_spatial_dims, num_points),
        ) + 1j * jr.normal(
            imag_key,
            shape=(1,) + wavenumber_shape(self.num_spatial_dims, num_points),
        )

        noise = noise * amplitude

        ic = jnp.fft.irfftn(
            noise,
            s=spatial_shape(self.num_spatial_dims, num_points),
            axes=space_indices(self.num_spatial_dims),
        )

        if self.zero_mean:
            ic = ic - jnp.mean(ic)

        if self.max_one:
            ic = ic / jnp.max(jnp.abs(ic))

        return ic
