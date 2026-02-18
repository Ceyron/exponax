import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .._spectral import (
    build_scaled_wavenumbers,
    fft,
    ifft,
)
from ._base_ic import (
    BaseRandomICGenerator,
    normalize_ic,
    validate_normalization_options,
)
from ._white_noise import WhiteNoise


class GaussianRandomField(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    powerlaw_exponent: float
    zero_mean: bool
    std_one: bool
    max_one: bool
    white_noise: WhiteNoise

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
        validate_normalization_options(
            zero_mean=zero_mean, std_one=std_one, max_one=max_one
        )
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.powerlaw_exponent = powerlaw_exponent
        self.zero_mean = zero_mean
        self.std_one = std_one
        self.max_one = max_one
        self.white_noise = WhiteNoise(num_spatial_dims)

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        noise = self.white_noise(num_points, key=key)

        noise_hat = fft(noise, num_spatial_dims=self.num_spatial_dims)

        wavenumber_grid = build_scaled_wavenumbers(
            self.num_spatial_dims, self.domain_extent, num_points
        )
        wavenumber_norm_grid = jnp.linalg.norm(wavenumber_grid, axis=0, keepdims=True)
        # Further division by 2.0 in the exponent is because we want to have the
        # **power-spectrum** follow a **power-law**. See
        # https://github.com/Ceyron/exponax/issues/9 for more details.
        amplitude = jnp.power(wavenumber_norm_grid, -self.powerlaw_exponent / 2.0)
        amplitude = (
            amplitude.flatten().at[0].set(1.0).reshape(wavenumber_norm_grid.shape)
        )

        noise_hat = noise_hat * amplitude

        ic = ifft(
            noise_hat, num_spatial_dims=self.num_spatial_dims, num_points=num_points
        )

        ic = normalize_ic(
            ic, zero_mean=self.zero_mean, std_one=self.std_one, max_one=self.max_one
        )

        return ic
