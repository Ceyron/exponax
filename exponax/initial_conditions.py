from abc import ABC, abstractmethod
from typing import List

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .sample_stepper import Diffusion
from .spectral import (
    build_scaled_wavenumbers,
    build_scaling_array,
    low_pass_filter_mask,
    space_indices,
    spatial_shape,
    wavenumber_shape,
)
from .utils import get_grid

# --- Base classes ---


class BaseIC(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, x: Float[Array, "D ... N"]) -> Float[Array, "1 ... N"]:
        """
        Evaluate the initial condition.

        **Arguments**:
            - `x`: The grid points.

        **Returns**:
            - `u`: The initial condition evaluated at the grid points.
        """
        pass


class BaseRandomICGenerator(eqx.Module):
    num_spatial_dims: int
    domain_extent: float
    indexing: str = "ij"

    def gen_ic_fun(self, num_points: int, *, key: PRNGKeyArray) -> BaseIC:
        """
        Generate an initial condition function.

        **Arguments**:
            - `num_points`: The number of grid points in each dimension.
            - `key`: A jax random key.

        **Returns**:
            - `ic`: An initial condition function that can be evaluated at
                degree of freedom locations.
        """
        raise NotImplementedError(
            "This random ic generator cannot represent its initial condition as a function. Directly evaluate it."
        )

    def __call__(
        self,
        num_points: int,
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "1 ... N"]:
        """
        Generate a random initial condition.

        **Arguments**:
            - `num_points`: The number of grid points in each dimension.
            - `key`: A jax random key.
            - `indexing`: The indexing convention for the grid.

        **Returns**:
            - `u`: The initial condition evaluated at the grid points.
        """
        ic_fun = self.gen_ic_fun(num_points, key=key)
        grid = get_grid(
            self.num_spatial_dims,
            self.domain_extent,
            num_points,
            indexing=self.indexing,
        )
        return ic_fun(grid)


# Utilities to create ICs for multi-channel fields


class MultiChannelIC(eqx.Module):
    initial_conditions: List[BaseIC]

    def __call__(self, x: Float[Array, "D ... N"]) -> Float[Array, "C ... N"]:
        """
        Evaluate the initial condition.

        **Arguments**:
            - `x`: The grid points.

        **Returns**:
            - `u`: The initial condition evaluated at the grid points.
        """
        return jnp.concatenate([ic(x) for ic in self.initial_conditions], axis=0)


class RandomMultiChannelICGenerator(eqx.Module):
    ic_generators: List[BaseRandomICGenerator]

    def gen_ic_fun(self, num_points: int, *, key: PRNGKeyArray) -> MultiChannelIC:
        ic_funs = [
            ic_gen.gen_ic_fun(num_points, key=key) for ic_gen in self.ic_generators
        ]
        return MultiChannelIC(ic_funs)

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "C ... N"]:
        u_list = [ic_gen(num_points, key=key) for ic_gen in self.ic_generators]
        return jnp.concatenate(u_list, axis=0)


# New version

# class TruncatedFourierSeries(BaseIC):
#     coefficient_array: Complex[Array, "1 ... (N//2)+1"]

#     def __init__(
#         self,
#         D: int,
#         L: float,  # unused
#         N: int,
#         *,
#         coefficient_array: Complex[Array, "1 ... N"],
#     ):
#         super().__init__(D, N)
#         self.coefficient_array = coefficient_array

#     def evaluate(self, x: Float[Array, "D ... N"]) -> Float[Array, "1 ... N"]:
#         return jnp.fft.irfftn(
#             self.coefficient_array,
#             s=spatial_shape(self.D, self.N),
#             axes=space_indices(self.D),
#         )


class RandomTruncatedFourierSeries(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    cutoff: int
    amplitude_range: tuple[int, int]
    angle_range: tuple[int, int]
    offset_range: tuple[int, int]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float = 1.0,
        *,
        cutoff: int = 10,
        amplitude_range: tuple[int, int] = (-1.0, 1.0),
        angle_range: tuple[int, int] = (0.0, 2.0 * jnp.pi),
        offset_range: tuple[int, int] = (0.0, 0.0),  # no offset by default
    ):
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent

        self.cutoff = cutoff
        self.amplitude_range = amplitude_range
        self.angle_range = angle_range
        self.offset_range = offset_range

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
            self.num_spatial_dims, num_points
        )

        u = jnp.fft.irfftn(
            fourier_noise,
            s=spatial_shape(self.num_spatial_dims, num_points),
            axes=space_indices(self.num_spatial_dims),
        )

        return u


# --- Legacy Sine Waves (truncated Fourier series) ---

# class SineWaves(BaseIC):
#     L: float
#     filter_mask: Float[Array, "1 ... (N//2)+1"]
#     zero_mean: bool
#     key: PRNGKeyArray


#     def __init__(
#         self,
#         D: int,
#         L: float,
#         N: int,
#         *,
#         cutoff: int,
#         zero_mean: bool,
#         axis_separate: bool = True,
#         key: PRNGKeyArray,
#     ):
#         super().__init__(D, N)
#         self.L = L
#         self.filter_mask = low_pass_filter_mask(D, N, cutoff=cutoff, axis_separate=axis_separate)
#         self.zero_mean = zero_mean
#         self.key = key

#     def evaluate(self, x: Float[Array, "D ... N"]) -> Float[Array, "1 ... N"]:
#         noise_shape = (1,) + spatial_shape(self.D, self.N)

#         noise = jr.normal(self.key, shape=noise_shape)
#         noise_hat = jnp.fft.rfftn(noise, axes=space_indices(self.D))
#         noise_hat = noise_hat * self.filter_mask

#         noise = jnp.fft.irfftn(noise_hat, s=spatial_shape(self.D, self.N), axes=space_indices(self.D))

#         if self.zero_mean:
#             noise = noise - jnp.mean(noise)

#         return noise

# class RandomSineWaves(BaseRandomICGenerator):
#     D: int
#     L: float
#     N: int
#     cutoff: int
#     zero_mean: bool
#     axis_separate: bool

#     def __init__(
#         self,
#         D: int,
#         L: float,
#         N: int,
#         *,
#         cutoff: int,
#         zero_mean: bool,
#         axis_separate: bool = True,
#     ):
#         """
#         Randomly generated initial condition consisting of a truncated Fourier series.

#         Arguments are drawn from uniform distributions.

#         **Arguments**:
#             - `D`: The dimension of the domain.
#             - `N`: The number of grid points in each dimension.
#             - `L`: The length of the domain.
#             - `cutoff`: The cutoff wavenumber.
#             - `zero_mean`: Whether to subtract the mean.
#             - `axis_separate`: Whether to draw the wavenumber cutoffs for each
#                 axis separately.
#         """
#         self.D = D
#         self.N = N
#         self.L = L
#         self.cutoff = cutoff
#         self.zero_mean = zero_mean
#         self.axis_separate = axis_separate

#     def __call__(self, key: PRNGKeyArray) -> SineWaves:
#         return SineWaves(
#             self.D,
#             self.L,
#             self.N,
#             cutoff=self.cutoff,
#             zero_mean=self.zero_mean,
#             axis_separate=self.axis_separate,
#             key=key,
#         )


# --- Diffused Noise --- ###

# class DiffusedNoise(BaseIC):
#     L: float
#     intensity: float
#     zero_mean: bool
#     key: PRNGKeyArray

#     def __init__(
#         self,
#         D: int,
#         L: float,
#         N: int,
#         *,
#         intensity: float,
#         zero_mean: bool,
#         key: PRNGKeyArray,
#     ):
#         super().__init__(D, N)
#         self.L = L
#         self.intensity = intensity
#         self.zero_mean = zero_mean
#         self.key = key

#     def evaluate(self, x: Float[Array, "D ... N"]) -> Float[Array, "1 ... N"]:
#         noise_shape = (1,) + spatial_shape(self.D, self.N)
#         noise = jr.normal(self.key, shape=noise_shape)

#         diffusion_stepper = Diffusion(self.D, self.L, self.N, 1.0, diffusivity=self.intensity)
#         ic = diffusion_stepper(noise)

#         if self.zero_mean:
#             ic = ic - jnp.mean(ic)

#         return ic


class DiffusedNoise(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    intensity: float
    zero_mean: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float = 1.0,
        *,
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


# Gausian Random Field


class GaussianRandomField(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    powerlaw_exponent: float
    normalize: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float = 1.0,
        *,
        powerlaw_exponent: float = 3.0,
        normalize: bool = True,
    ):
        """
        Randomly generated initial condition consisting of a Gaussian random field.
        """
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.powerlaw_exponent = powerlaw_exponent
        self.normalize = normalize

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

        if self.normalize:
            ic = ic - jnp.mean(ic)
            ic = ic / jnp.std(ic)

        return ic


# Discontinuities


class Discontinuities(BaseIC):
    pass


class RandomDiscontinuities(BaseRandomICGenerator):
    pass
