import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .._spectral import (
    build_scaled_wavenumbers,
    build_scaling_array,
    ifft,
    wavenumber_shape,
)
from ._base_ic import BaseRandomICGenerator


class GaussianRandomField(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    powerlaw_exponent: float
    zero_mean: bool
    std_one: bool
    max_one: bool

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        domain_extent: float = 1.0,
        powerlaw_exponent: float = 3.0,
        zero_mean: bool = True,
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        Random generator for initial states following a power-law spectrum in
        Fourier space, i.e., it decays polynomially with the wavenumber.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions.
        - `domain_extent`: The extent of the domain in each spatial direction.
        - `powerlaw_exponent`: The exponent of the power-law spectrum.
        - `zero_mean`: Whether the field should have zero mean.
        - `std_one`: Whether to normalize the state to have a standard
            deviation of one. Defaults to `False`. Only works if the offset is
            zero.
        - `max_one`: Whether to normalize the state to have the maximum
            absolute value of one. Defaults to `False`. Only one of `std_one`
            and `max_one` can be `True`.
        """
        if not zero_mean and std_one:
            raise ValueError("Cannot have `zero_mean=False` and `std_one=True`.")
        if std_one and max_one:
            raise ValueError("Cannot have `std_one=True` and `max_one=True`.")
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.powerlaw_exponent = powerlaw_exponent
        self.zero_mean = zero_mean
        self.std_one = std_one
        self.max_one = max_one

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        wavenumber_grid = build_scaled_wavenumbers(
            self.num_spatial_dims, self.domain_extent, num_points
        )
        wavenumer_norm_grid = jnp.linalg.norm(wavenumber_grid, axis=0, keepdims=True)
        # Further division by 2.0 in the exponent is because we want to have the
        # **power-spectrum** follow a **power-law**. See
        # https://github.com/Ceyron/exponax/issues/9 for more details.
        amplitude = jnp.power(wavenumer_norm_grid, -self.powerlaw_exponent / 2.0)
        amplitude = (
            amplitude.flatten().at[0].set(1.0).reshape(wavenumer_norm_grid.shape)
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

        noise = noise * build_scaling_array(
            self.num_spatial_dims, num_points, mode="coef_extraction"
        )

        ic = ifft(noise, num_spatial_dims=self.num_spatial_dims, num_points=num_points)

        if self.zero_mean:
            ic = ic - jnp.mean(ic)

        if self.std_one:
            ic = ic / jnp.std(ic)

        if self.max_one:
            ic = ic / jnp.max(jnp.abs(ic))

        return ic
