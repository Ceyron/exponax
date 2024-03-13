from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex

# E can either be 1 (single channel) or num_channels (multi-channel) for either
# the same linear operator for each channel or a different linear operator for
# each channel, respectively.
#
# So far, we do **not** support channel mixing via the linear operator (for
# example if we solved the wave equation or the sine-Gordon equation).


class BaseETDRK(eqx.Module, ABC):
    dt: float
    _exp_term: Complex[Array, "E ... (N//2)+1"]

    def __init__(
        self,
        dt: float,
        linear_operator: Complex[Array, "E ... (N//2)+1"],
    ):
        self.dt = dt
        self._exp_term = jnp.exp(self.dt * linear_operator)

    @abstractmethod
    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Advance the state in Fourier space.
        """
        pass
