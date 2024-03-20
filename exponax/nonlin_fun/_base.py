from abc import ABC, abstractmethod
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float

from .._spectral import low_pass_filter_mask, space_indices, spatial_shape


class BaseNonlinearFun(eqx.Module, ABC):
    num_spatial_dims: int
    num_points: int
    dealiasing_mask: Optional[Bool[Array, "1 ... (N//2)+1"]]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: Optional[float] = None,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.num_points = num_points

        if dealiasing_fraction is None:
            self.dealiasing_mask = None
        else:
            # Can be done because num_points is identical in all spatial dimensions
            nyquist_mode = (num_points // 2) + 1
            highest_resolved_mode = nyquist_mode - 1
            start_of_aliased_modes = dealiasing_fraction * highest_resolved_mode

            self.dealiasing_mask = low_pass_filter_mask(
                num_spatial_dims,
                num_points,
                cutoff=start_of_aliased_modes - 1,
            )

    def dealias(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        if self.dealiasing_mask is None:
            raise ValueError("Nonlinear function was set up without dealiasing")
        return self.dealiasing_mask * u_hat

    def fft(self, u: Float[Array, "C ... N"]) -> Complex[Array, "C ... (N//2)+1"]:
        return jnp.fft.rfftn(u, axes=space_indices(self.num_spatial_dims))

    def ifft(self, u_hat: Complex[Array, "C ... (N//2)+1"]) -> Float[Array, "C ... N"]:
        return jnp.fft.irfftn(
            u_hat,
            s=spatial_shape(self.num_spatial_dims, self.num_points),
            axes=space_indices(self.num_spatial_dims),
        )

    @abstractmethod
    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Evaluate all potential nonlinearities "pseudo-spectrally", account for dealiasing.
        """
        pass
