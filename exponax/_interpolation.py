"""
Utilities to map Exponax states to different grids.
"""
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._spectral import (
    build_reconstructional_scaling_array,
    build_scaled_wavenumbers,
    fft,
    space_indices,
)

C = TypeVar("C")  # Channel axis
D = TypeVar(
    "D"
)  # Dimension axis - must have as many dimensions as the array has subsequent spatial axes
N = TypeVar("N")  # Spatial axis


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
        indexing: str = "ij",
    ):
        """
        Assumes that the indexing convention is "ij"
        """
        self.num_spatial_dims = state.ndim - 1
        self.domain_extent = domain_extent
        self.num_points = state.shape[-1]

        self.state_hat_scaled = fft(state, num_spatial_dims=self.num_spatial_dims) / (
            build_reconstructional_scaling_array(
                self.num_spatial_dims, self.num_points, indexing=indexing
            )
        )
        self.wavenumbers = build_scaled_wavenumbers(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            indexing=indexing,
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

        # The exponential term sums over the wavenumber dimension axis (`"D"`)
        exp_term: Complex[Array, "... (N//2)+1"] = jnp.exp(
            jnp.sum(1j * self.wavenumbers * x_bloated, axis=0)
        )

        # Re-add a singleton channel axis to have broadcasting work correctly
        exp_term: Complex[Array, "1 ... (N//2)+1"] = exp_term[None, ...]

        interpolation_operation: Complex[Array, "C ... (N//2)+1"] = (
            self.state_hat_scaled * exp_term
        )

        interpolated_value: Float[Array, "C"] = jnp.real(
            jax.vmap(jnp.sum)(interpolation_operation)
        )

        return interpolated_value
