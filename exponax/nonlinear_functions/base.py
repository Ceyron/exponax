import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Complex, Array, Float, Bool
from ..spectral import (
    wavenumber_shape,
    low_pass_filter_mask,
)
from abc import ABC, abstractmethod


class BaseNonlinearFun(eqx.Module, ABC):
    num_spatial_dims: int
    num_points: int
    num_channels: int
    derivative_operator: Complex[Array, "D ... (N//2)+1"]
    dealiasing_mask: Bool[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.num_points = num_points
        self.num_channels = num_channels
        self.derivative_operator = derivative_operator

        # Can be done because num_points is identical in all spatial dimensions
        nyquist_mode = (num_points // 2) + 1
        highest_resolved_mode = nyquist_mode - 1
        start_of_aliased_modes = dealiasing_fraction * highest_resolved_mode

        self.dealiasing_mask = low_pass_filter_mask(
            num_spatial_dims,
            num_points,
            cutoff=start_of_aliased_modes - 1,
        )

    @abstractmethod
    def evaluate(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Evaluate all potential nonlinearities "pseudo-spectrally", account for dealiasing.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Perform check
        """
        expected_shape = (self.num_channels,) + wavenumber_shape(
            self.num_spatial_dims, self.num_points
        )
        if u_hat.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {u_hat.shape}. For batched operation use `jax.vmap` on this function."
            )

        return self.evaluate(u_hat)
