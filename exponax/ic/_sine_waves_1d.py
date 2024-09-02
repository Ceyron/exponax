import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from ._base_ic import BaseIC, BaseRandomICGenerator


class SineWaves1d(BaseIC):
    domain_extent: float
    amplitudes: tuple[float, ...]
    wavenumbers: tuple[float, ...]
    phases: tuple[float, ...]
    offset: float

    std_one: bool
    max_one: bool

    def __init__(
        self,
        domain_extent: float,
        amplitudes: tuple[float, ...],
        wavenumbers: tuple[float, ...],
        phases: tuple[float, ...],
        offset: float = 0.0,
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        A state described by a collection of sine waves. Only works in 1d.

        **Arguments**:

        - `domain_extent`: The extent of the domain.
        - `amplitudes`: A tuple of amplitudes.
        - `wavenumbers`: A tuple of wavenumbers.
        - `phases`: A tuple of phases.
        - `offset`: A constant offset.
        - `std_one`: Whether to normalize the state to have a standard
            deviation of one. Defaults to `False`. Only works if the offset
            is zero.
        - `max_one`: Whether to normalize the state to have the maximum
            absolute value of one. Defaults to `False`. Only one of
            `std_one` and `max_one` can be `True`.
        """
        if offset != 0.0 and std_one:
            raise ValueError("Cannot have non-zero offset and `std_one=True`.")
        if std_one and max_one:
            raise ValueError("Cannot have `std_one=True` and `max_one=True`.")

        if len(amplitudes) != len(wavenumbers) or len(wavenumbers) != len(phases):
            raise ValueError(
                "The number of amplitudes, wavenumbers, and phases must be the same."
            )

        self.domain_extent = domain_extent
        self.amplitudes = amplitudes
        self.wavenumbers = wavenumbers
        self.phases = phases
        self.offset = offset
        self.std_one = std_one
        self.max_one = max_one

    def __call__(self, x: Float[Array, "1 N"]) -> Float[Array, "1 N"]:
        if x.shape[0] != 1:
            raise ValueError("SineWaves1d only works in 1d.")
        result = jnp.zeros_like(x)
        for a, k, p in zip(self.amplitudes, self.wavenumbers, self.phases):
            result += a * jnp.sin(k * (2 * jnp.pi / self.domain_extent) * x + p)
        result += self.offset

        if self.std_one:
            result = result / jnp.std(result)

        if self.max_one:
            result = result / jnp.max(jnp.abs(result))

        return result


class RandomSineWaves1d(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    cutoff: int
    amplitude_range: tuple[float, float]
    phase_range: tuple[float, float]
    offset_range: tuple[float, float]

    std_one: bool
    max_one: bool

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        domain_extent: float = 1.0,
        cutoff: int = 5,
        amplitude_range: tuple[float, float] = (-1.0, 1.0),
        phase_range: tuple[float, float] = (0.0, 2 * jnp.pi),
        offset_range: tuple[float, float] = (0.0, 0.0),
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        Random generator for initial states described by a collection of sine
        waves. Only works in 1d.

        This is a simplified version of the `RandomTruncatedFourierSeries`
        generator that works in arbitrary dimensions. However, only this
        generator can produce a functional representation of the initial
        condition.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions.
        - `domain_extent`: The extent of the domain.
        - `cutoff`: The cutoff of the wavenumbers. This limits the
            "complexity" of the initial state. Note that some dynamics are very
            sensitive to high-frequency information.
        - `amplitude_range`: The range of the amplitudes. Defaults to
            `(-1.0, 1.0)`.
        - `phase_range`: The range of the phases. Defaults to `(0.0, 2π)`.
        - `offset_range`: The range of the offsets. Defaults to `(0.0,
            0.0)`, meaning **zero-mean** by default.
        - `std_one`: Whether to normalize the state to have a standard
            deviation of one. Defaults to `False`. Only works if the offset is
            zero.
        - `max_one`: Whether to normalize the state to have the maximum
            absolute value of one. Defaults to `False`. Only one of `std_one`
            and `max_one` can be `True`.
        """
        if num_spatial_dims != 1:
            raise ValueError("RandomSineWaves1d only works in 1d.")
        if offset_range != (0.0, 0.0) and std_one:
            raise ValueError("Cannot have non-zero offset and `std_one=True`.")
        if std_one and max_one:
            raise ValueError("Cannot have `std_one=True` and `max_one=True`.")

        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.cutoff = cutoff
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.offset_range = offset_range
        self.std_one = std_one
        self.max_one = max_one

    def gen_ic_fun(self, *, key: PRNGKeyArray) -> SineWaves1d:
        amplitude_key, phase_key, offset_key = jr.split(key, 3)

        amplitudes = jr.uniform(
            amplitude_key,
            shape=(self.cutoff,),
            minval=self.amplitude_range[0],
            maxval=self.amplitude_range[1],
        )
        phases = jr.uniform(
            phase_key,
            shape=(self.cutoff,),
            minval=self.phase_range[0],
            maxval=self.phase_range[1],
        )
        offset = jr.uniform(
            offset_key,
            shape=(),
            minval=self.offset_range[0],
            maxval=self.offset_range[1],
        )

        return SineWaves1d(
            domain_extent=self.domain_extent,
            amplitudes=amplitudes,
            wavenumbers=jnp.arange(1, self.cutoff + 1),
            phases=phases,
            offset=offset,
            std_one=self.std_one,
            max_one=self.max_one,
        )
