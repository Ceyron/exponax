"""
Utilities to map Exponax states to different grids.
"""
from typing import Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._spectral import (
    build_scaled_wavenumbers,
    build_scaling_array,
    fft,
    get_modes_slices,
    ifft,
    oddball_filter_mask,
    space_indices,
    wavenumber_shape,
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
        indexing: Literal["ij", "xy"] = "ij",
    ):
        """
        Builds an interpolation function for an `Exponax` state using its
        Fourier representation.

        After instantiation, the interpolant can be called with a query
        coordinate `x ∈ ℝᴰ` (e.g., `x = jnp.array([0.3, 0.5])` in 2D) to obtain
        the corresponding value. If the query coordinate is not within the
        domain, i.e., `x ∉ Ω = [0, L]ᴰ`, the returned result is found in its
        periodic extension.

        !!! info
            If the state is band-limited, i.e., the highest wavenumber
            containing non-zero energy is at max `(N//2)`, then the
            interpolation will be exact (no interpolation error).

        !!! warning
            This interpolation uses global basis functions. Hence its memory and
            computation for evaluating one query location scales with `O(N^D)`.
            Consequently, if multiple query locations are to be evaluated in
            parallel (via `jax.vmap`), the memory and computation scales with
            `O(N^D * M)` where `M` is the number of query locations. This can
            easily exceed available resources. In such cases, either consider
            evaluating the query locations in smaller batches or resort to local
            basis interpolants like linear or cubic splines (see
            `scipy.interpolate` or its JAX anologons).

        **Arguments:**

        - `state`: The state to interpolate. Must conform to the `Exponax`
            standard with a leading channel axis (can be a singleton axis if
            there is only one channel), and one, two, or three subsequent
            spatial axes (depending on the number of spatial dimensions). These
            latter spatial axes must have the same number of dimensions.
        - `domain_extent`: The size of the domain `L`; in higher dimensions the
            domain is assumed to be a scaled hypercube `Ω = (0, L)ᴰ`.
        - `indexing`: The indexing convention of the spatial axes. The default
            `"ij"` follows the `Exponax` convention.
        """
        self.num_spatial_dims = state.ndim - 1
        self.domain_extent = domain_extent
        self.num_points = state.shape[-1]

        self.state_hat_scaled = fft(state, num_spatial_dims=self.num_spatial_dims) / (
            build_scaling_array(
                self.num_spatial_dims,
                self.num_points,
                mode="reconstruction",
                indexing=indexing,
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
        Evaluate the interpolant at the query location `x`.

        **Arguments:**

        - `x`: The query location. Must be a vector of length `D` where `D` is
            the number of spatial dimensions. This must match the number of
            spatial dimensions of the state used to build the interpolant.

        **Returns:**

        - `interpolated_value`: The interpolated value at the query location
            `x`. This will have as many channels as the state used to build the
            interpolant.


        !!! tip
            To evaluate the interpolant at multiple query locations in parallel,
            use `jax.vmap`. For example, in 1d:

            ```python

            print(state.shape)  # (C, N)

            interpolator = FourierInterpolator(state, domain_extent=1.0)

            print(query_locations.shape)  # (1, M)

            interpolated_values = jax.vmap(
                interpolator, in_axes=-1, out_axes=-1,
            )(query_locations)

            print(interpolated_values.shape)  # (C, M)

            ```

            If the query locations have multiple batch axes (e.g., to represent
            another grid), consider using nested `jax.vmap` calls. For example,
            in 2D

            ```python

            print(state.shape)  # (C, N, N)

            interpolator = FourierInterpolator(state, domain_extent=1.0)

            print(query_locations.shape)  # (2, M, P)

            interpolated_values = jax.vmap(
                jax.vmap(interpolator, in_axes=-1, out_axes=-1), in_axes=-2,
                out_axes=-2,
            )(query_locations)

            print(interpolated_values.shape)  # (C, M, P)

            ```

        !!! warning
            This interpolation uses global basis functions. Hence its memory and
            computation for evaluating one query location scales with `O(N^D)`.
            Consequently, if multiple query locations are to be evaluated in
            parallel (via `jax.vmap`), the memory and computation scales with
            `O(N^D * M)` where `M` is the number of query locations. This can
            easily exceed available resources. In such cases, consider
            evaluating the query locations in smaller batches.
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


def map_between_resolutions(
    state: Float[Array, "C ... N"],
    new_num_points: int,
    *,
    oddball_zero: bool = True,
) -> Float[Array, "C ... N_new"]:
    """
    Upsamples or downsamples a state in `Exponax` convention to a new resolution
    via manipulation of its Fourier representation.

    This approach is way more efficient that `exponax.FourierInterpolator` but
    can only move the state between uniform Cartesian grids of different
    resolutions.

    !!! info
        If the new resolution is higher than the old resolution, the state is
        upsampled. If the new resolution is lower than the old resolution, the
        state is downsampled. If the given state is bandlimited, i.e., the
        highest wavenumber containing non-zero energy is at max `(N//2)`, then
        upsampling will be exact (no interpolation error). Also, in case of
        downsampling: if the given state was bandlimited, and the it would be
        still be bandlimited in the new resolution, this downsampling will also
        be exact, i.e., no coarsening artifacts. If this is not the case, one
        loses high-frequency (fine scale) information.

    **Arguments:**

    - `state`: The state to interpolate. Must conform to the `Exponax`
        standard with a leading channel axis (can be a singleton axis if there
        is only one channel), and one, two, or three subsequent spatial axes
        (depending on the number of spatial dimensions). These latter spatial
        axes must have the same number of dimensions.
    - `new_num_points`: The new number of points in each spatial dimension.
    - `oddball_zero`: Whether to zero out the Nyquist frequency in case of
        even-sized grids. This is usually preferred.

    **Returns:**

    - `new_state`: The state interpolated to the new resolution. This will have
        the same number of channels as the input state.
    """
    num_spatial_dims = state.ndim - 1
    old_num_points = state.shape[-1]
    num_channels = state.shape[0]

    if old_num_points == new_num_points:
        return state

    old_state_hat_scaled = fft(
        state, num_spatial_dims=num_spatial_dims
    ) / build_scaling_array(
        num_spatial_dims,
        old_num_points,
        mode="norm_compensation",
    )

    if new_num_points > old_num_points:
        # Upscaling
        if old_num_points % 2 == 0 and oddball_zero:
            old_state_hat_scaled *= oddball_filter_mask(
                num_spatial_dims, old_num_points
            )

    new_state_hat_scaled = jnp.zeros(
        (num_channels,) + wavenumber_shape(num_spatial_dims, new_num_points),
        dtype=old_state_hat_scaled.dtype,
    )

    modes_slices: list[list[slice]] = get_modes_slices(
        num_spatial_dims,
        min(old_num_points, new_num_points),
    )

    for block_slice in modes_slices:
        new_state_hat_scaled = new_state_hat_scaled.at[block_slice].set(
            old_state_hat_scaled[block_slice]
        )

    new_state_hat = new_state_hat_scaled * build_scaling_array(
        num_spatial_dims,
        new_num_points,
        mode="norm_compensation",
    )
    if old_num_points > new_num_points:
        # Downscaling
        if new_num_points % 2 == 0 and oddball_zero:
            new_state_hat *= oddball_filter_mask(num_spatial_dims, new_num_points)

    new_state = ifft(
        new_state_hat,
        num_spatial_dims=num_spatial_dims,
        num_points=new_num_points,
    )

    return new_state
