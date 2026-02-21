from typing import Sequence
import jax.numpy as jnp
from jaxtyping import Array, Complex
from ._base import BaseNonlinearFun


class TamedPolynomialNonlinearFun(BaseNonlinearFun):
    """
    Pseudo-spectral polynomial nonlinear function with optional Hutzenthaler-Jentzen taming.

    Evaluates the polynomial
        p(u) = sum_{i=0}^{p-1} coeffs[i] * u**i
    pointwise in physical space (after pre-dealiasing via self.ifft), optionally
    applies the taming denominator 1/(1 + dt * |p(u)|), and returns the
    post-dealiased Fourier transform.

    Parameters
    ----------
    num_spatial_dims, num_points, dealiasing_fraction:
        forwarded to BaseNonlinearFun (controls fft/ifft and dealias mask).
    coefficients:
        Sequence of polynomial coefficients [c0, c1, c2, ...] such that
        p(u) = c0 + c1 u + c2 u^2 + ...
    dt:
        Time step (used only when use_taming=True).
    use_taming:
        If True apply N_tamed = N / (1 + dt * |N|).

    Notes
    -----
    - The Allen-Cahn cubic (−λ u^3) is represented by coefficients = [0, 0, 0, -λ].
    """
    coefficients: Sequence[float]
    dt: float
    use_taming: bool

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: float,
        coefficients: Sequence[float],
        dt: float,
        use_taming: bool = True,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        # keep python sequence for serialisation; convert to jnp array on use
        self.coefficients = tuple(float(c) for c in coefficients)
        self.dt = float(dt)
        self.use_taming = bool(use_taming)

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        # pre-dealias + to physical
        # physical-space real array (shape (C, ... , *spatial))
        u = self.ifft(u_hat)

        # evaluate polynomial p(u) = sum coeffs[i] * u**i
        # start with c0 (constant)
        nonlin = jnp.zeros_like(u)
        # compute powers progressively to keep cost smaller than repeated ** calls
        u_power = jnp.ones_like(u)  # u**0
        for c in self.coefficients:
            if c != 0.0:
                nonlin = nonlin + c * u_power
            u_power = u_power * u  # increment power for next iteration

        # apply taming if requested: N_tamed = N / (1 + dt * |N|)
        if self.use_taming:
            denom = 1.0 + self.dt * jnp.abs(nonlin)
            nonlin = nonlin / denom

        # forward transform and post-dealias
        return self.fft(nonlin)
