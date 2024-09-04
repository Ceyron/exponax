"""
Utilities to map Exponax states to different grids.
"""
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._spectral import build_scaled_wavenumbers, build_scaling_array, fft, space_indices

C = TypeVar("C")  # Channel axis
D = TypeVar(
    "D"
)  # Dimension axis - must have as many dimensions as the array has subsequent spatial axes


class FourierInterpolator(eqx.Module):
    num_spatial_dims: int
    domain_extent: float
    num_points: int
    state_hat_scaled: Complex[Array, "C ... (N//2)+1"]
    wavenumbers: Float[Array, "D ... (N//2)+1"]

    def __init__(
        self,
        state: Float[Array, "C ... N"],
        *,
        domain_extent: float = 1.0,
    ):
        """
        Assumes that the indexing convention is "ij"
        """
        self.num_spatial_dims = state.ndim - 1
        self.domain_extent = domain_extent
        self.num_points = state.shape[-1]

        self.state_hat_scaled = fft(
            state, num_spatial_dims=self.num_spatial_dims
        ) / build_scaling_array(self.num_spatial_dims, self.num_points)
        self.wavenumbers = build_scaled_wavenumbers(
            self.num_spatial_dims, self.domain_extent, self.num_points
        )

    def __call__(
        self,
        x: Float[Array, "D"],
    ) -> Float[Array, "C"]:
        """
        use `jax.vmap(..., axis=(-1))` on this for batched operation
        """
        # Adds singleton axes for each spatial dimension
        x_bloated: Float[Array, "D ... 1"] = jnp.expand_dims(
            x, axis=space_indices(self.num_spatial_dims)
        )
        # Adds singleton axis for channels
        x_bloated: Float[Array, "1 D ... 1"] = x_bloated[None]

        # Add singleton axis at position where `x_bloated` has its spatial axis
        # D
        state_hat_scaled_bloated: Complex[
            Array, "C 1 ... (N//2)+1"
        ] = self.state_hat_scaled[:, None]

        # Add singleton axis for channels
        wavenumbers_bloated: Float[Array, "1 D ... (N//2)+1"] = self.wavenumbers[None]

        interpolation_operation: Complex[
            Array, "C D ... (N//2)+1"
        ] = state_hat_scaled_bloated * jnp.exp(1j * wavenumbers_bloated * x_bloated)

        interpolated_value: Float[Array, "C"] = jnp.real(
            jax.vmap(jnp.sum)(interpolation_operation)
        )

        return interpolated_value
