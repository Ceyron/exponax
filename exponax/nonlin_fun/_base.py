from abc import ABC, abstractmethod
from typing import Optional

import equinox as eqx
from jaxtyping import Array, Bool, Complex, Float

from .._spectral import fft, ifft, low_pass_filter_mask


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
        """
        Base class for all nonlinear functions. This class provides the basic
        functionality to dealias the nonlinear terms and perform forward and
        inverse Fourier transforms.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same.
        - `dealiasing_fraction`: The fraction of the highest resolved mode to
            keep for dealiasing. For example, `2/3` corresponds to Orszag's 2/3
            rule typically used for quadratic nonlinearities. If `None`, no
            dealiasing is performed.

        !!! info
            Some dealiasing strategies (like Orszag's 2/3 rule) are designed to
            not fully remove aliasing (which would require 1/2 in the case of
            quadratic nonlinearities), rather to only have aliases being created
            in those modes that will be zeroed out anyway in the next
            dealiasing step. See also [Orszag
            (1971)](https://doi.org/10.1175/1520-0469(1971)028%3C1074:OTEOAI%3E2.0.CO;2)
        """
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
        """
        Dealias the Fourier representation of a state `u_hat` by zeroing out all
        the coefficients associated with modes beyond `dealiasing_fraction` set
        in the constructor.

        **Arguments:**

        - `u_hat`: The Fourier representation of the state `u`.

        **Returns:**

        - `u_hat_dealiased`: The dealiased Fourier representation of the state
            `u`.
        """
        if self.dealiasing_mask is None:
            raise ValueError("Nonlinear function was set up without dealiasing")
        return self.dealiasing_mask * u_hat

    def fft(self, u: Float[Array, "C ... N"]) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Correctly wrapped **real-valued** Fourier transform for the shape of the
        state vector associated with this nonlinear function.

        **Arguments:**

        - `u`: The state vector in real space.

        **Returns:**

        - `u_hat`: The (real-valued) Fourier transform of the state vector.
        """
        return fft(u, num_spatial_dims=self.num_spatial_dims)

    def ifft(self, u_hat: Complex[Array, "C ... (N//2)+1"]) -> Float[Array, "C ... N"]:
        """
        Correctly wrapped **real-valued** inverse Fourier transform for the shape
        of the state vector associated with this nonlinear function.

        **Arguments:**

        - `u_hat`: The (real-valued) Fourier transform of the state vector.

        **Returns:**

        - `u`: The state vector in real space.
        """
        return ifft(
            u_hat, num_spatial_dims=self.num_spatial_dims, num_points=self.num_points
        )

    @abstractmethod
    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Evaluates the nonlinear function with a pseudo-spectral treatment and
        accounts for dealiasing.

        Use this in combination with `exponax.etdrk` routines to solve
        semi-linear PDEs in Fourier space.

        **Arguments:**

        - `u_hat`: The Fourier representation of the state `u`.

        **Returns:**

        - `𝒩(u_hat)`: The Fourier representation of the nonlinear term.
        """
        pass
