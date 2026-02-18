import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .._spectral import (
    fft,
    ifft,
    low_pass_filter_mask,
)
from ._base_ic import (
    BaseRandomICGenerator,
    normalize_ic,
    validate_normalization_options,
)
from ._white_noise import WhiteNoise


class RandomTruncatedFourierSeries(BaseRandomICGenerator):
    num_spatial_dims: int
    cutoff: int
    offset_range: tuple[int, int]
    std_one: bool
    max_one: bool
    white_noise: WhiteNoise

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        cutoff: int = 5,
        offset_range: tuple[int, int] = (0.0, 0.0),  # no offset by default
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        Random generator for initial states consisting of a truncated Fourier
        series. White noise is drawn in physical space, transformed to Fourier
        space, low-pass filtered up to ``cutoff``, and transformed back.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `cutoff`: The cutoff of the wavenumbers. This limits the
            "complexity" of the initial state. Note that some dynamics are very
            sensitive to high-frequency information.
        - `offset_range`: The range of the offsets. Defaults to `(0.0,
            0.0)`, meaning **zero-mean** by default.
        - `std_one`: Whether to normalize the state to have a standard
            deviation of one. Defaults to `False`. Only works if the offset is
            zero.
        - `max_one`: Whether to normalize the state to have the maximum
            absolute value of one. Defaults to `False`. Only one of `std_one`
            and `max_one` can be `True`.
        """
        zero_mean = offset_range == (0.0, 0.0)
        validate_normalization_options(
            zero_mean=zero_mean, std_one=std_one, max_one=max_one
        )
        self.num_spatial_dims = num_spatial_dims
        self.cutoff = cutoff
        self.offset_range = offset_range
        self.std_one = std_one
        self.max_one = max_one
        self.white_noise = WhiteNoise(num_spatial_dims)

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        noise_key, offset_key = jr.split(key)

        noise = self.white_noise(num_points, key=noise_key)

        noise_hat = fft(noise, num_spatial_dims=self.num_spatial_dims)

        low_pass_filter = low_pass_filter_mask(
            self.num_spatial_dims, num_points, cutoff=self.cutoff, axis_separate=True
        )

        noise_hat = noise_hat * low_pass_filter

        offset = jr.uniform(
            offset_key,
            shape=(1,),
            minval=self.offset_range[0],
            maxval=self.offset_range[1],
        )[0]
        fourier_noise_shape = noise_hat.shape
        noise_hat = noise_hat.flatten().at[0].set(offset).reshape(fourier_noise_shape)

        ic = ifft(
            noise_hat,
            num_spatial_dims=self.num_spatial_dims,
            num_points=num_points,
        )

        zero_mean = self.offset_range == (0.0, 0.0)
        ic = normalize_ic(
            ic, zero_mean=zero_mean, std_one=self.std_one, max_one=self.max_one
        )

        return ic
