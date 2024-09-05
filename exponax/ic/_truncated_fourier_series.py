import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .._spectral import (
    build_scaling_array,
    ifft,
    low_pass_filter_mask,
    wavenumber_shape,
)
from ._base_ic import BaseRandomICGenerator


class RandomTruncatedFourierSeries(BaseRandomICGenerator):
    num_spatial_dims: int
    cutoff: int
    amplitude_range: tuple[int, int]
    angle_range: tuple[int, int]
    offset_range: tuple[int, int]
    std_one: bool
    max_one: bool

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        cutoff: int = 5,
        amplitude_range: tuple[int, int] = (-1.0, 1.0),
        angle_range: tuple[int, int] = (0.0, 2.0 * jnp.pi),
        offset_range: tuple[int, int] = (0.0, 0.0),  # no offset by default
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        Random generator for initial states consisting of a truncated Fourier
        series with random Fourier coefficients.

        In 1d, the functional form reads:

        ```
            u(x) = o + ∑ₖ aₖ sin(k (2π/L) x) + bₖ cos(k (2 π)/L x)
        ```

        where `o` is the offset, `aₖ` and `bₖ` are the amplitudes of the sine
        and cosine terms, respectively, and `k` is the wavenumber which ranges
        up to `cutoff`. An equivalent representation is via angular offsets

        ```
            u(x) = o + ∑ₖ aₖ sin(k (2π/L) x + ϕₖ)
        ```

        where `ϕₖ` is the angular offset.

        The generalization to higher dimensions includes mixed terms and is not
        that straightforward to write down.

        Offsets are drawn accoriding to a uniform distribution in the range
        `offset_range`. Amplitudes are drawn according to a uniform distribution
        in the range `amplitude_range`. Angles (=angular offsets) are drawn
        according to a uniform distribution in the range `angle_range`.

        See also `exponax.ic.RandomSineWaves1d` for a simplified version that
        only works in 1d but can also produce a functional representation of the
        initial state.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `cutoff`: The cutoff of the wavenumbers. This limits the
            "complexity" of the initial state. Note that some dynamics are very
            sensitive to high-frequency information.
        - `amplitude_range`: The range of the amplitudes. Defaults to
            `(-1.0, 1.0)`.
        - `angle_range`: The range of the angles. Defaults to `(0.0, 2π)`.
        - `offset_range`: The range of the offsets. Defaults to `(0.0,
            0.0)`, meaning **zero-mean** by default.
        - `std_one`: Whether to normalize the state to have a standard
            deviation of one. Defaults to `False`. Only works if the offset is
            zero.
        - `max_one`: Whether to normalize the state to have the maximum
            absolute value of one. Defaults to `False`. Only one of `std_one`
            and `max_one` can be `True`.
        """
        if offset_range == (0.0, 0.0) and std_one:
            raise ValueError("Cannot have non-zero offset and `std_one=True`.")
        if std_one and max_one:
            raise ValueError("Cannot have `std_one=True` and `max_one=True`.")
        self.num_spatial_dims = num_spatial_dims

        self.cutoff = cutoff
        self.amplitude_range = amplitude_range
        self.angle_range = angle_range
        self.offset_range = offset_range
        self.std_one = std_one
        self.max_one = max_one

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        fourier_noise_shape = (1,) + wavenumber_shape(self.num_spatial_dims, num_points)
        amplitude_key, angle_key, offset_key = jr.split(key, 3)

        amplitude = jr.uniform(
            amplitude_key,
            shape=fourier_noise_shape,
            minval=self.amplitude_range[0],
            maxval=self.amplitude_range[1],
        )
        angle = jr.uniform(
            angle_key,
            shape=fourier_noise_shape,
            minval=self.angle_range[0],
            maxval=self.angle_range[1],
        )

        fourier_noise = amplitude * jnp.exp(1j * angle)

        low_pass_filter = low_pass_filter_mask(
            self.num_spatial_dims, num_points, cutoff=self.cutoff, axis_separate=True
        )

        fourier_noise = fourier_noise * low_pass_filter

        offset = jr.uniform(
            offset_key,
            shape=(1,),
            minval=self.offset_range[0],
            maxval=self.offset_range[1],
        )[0]
        fourier_noise = (
            fourier_noise.flatten().at[0].set(offset).reshape(fourier_noise_shape)
        )

        fourier_noise = fourier_noise * build_scaling_array(
            self.num_spatial_dims,
            num_points,
            mode="coef_extraction",
        )

        u = ifft(
            fourier_noise,
            num_spatial_dims=self.num_spatial_dims,
            num_points=num_points,
        )

        if self.std_one:
            u = u / jnp.std(u)

        if self.max_one:
            u /= jnp.max(jnp.abs(u))

        return u
