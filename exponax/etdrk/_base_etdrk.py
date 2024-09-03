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
        """
        Base class for exponential time differencing Runge-Kutta methods.

        **Arguments:**

        - `dt`: The time step size.
        - `linear_operator`: The linear operator of the PDE. Must have a leading
            channel axis, followed by one, two or three spatial axes whereas the
            last axis must be of size `(N//2)+1` where `N` is the number of
            dimensions in the former spatial axes.

        !!! Example
            Below is an example how to get the linear operator for
            the heat equation.

            ```python
            import jax.numpy as jnp
            import exponax as ex

            # Define the linear operator
            N = 256
            L = 5.0  # The domain size
            D = 1  # Being in 1D

            derivative_operator = 1j * ex.spectral.build_derivative_operator(
                D,
                L,
                N,
            )

            print(derivative_operator.shape)  # (1, (N//2)+1)

            nu = 0.01 # The diffusion coefficient

            linear_operator = nu * derivative_operator**2
            ```
        """
        self.dt = dt
        self._exp_term = jnp.exp(self.dt * linear_operator)

    @abstractmethod
    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Advance the state in Fourier space.

        **Arguments:**

        - `u_hat`: The previous state in Fourier space.

        **Returns:**

        - The next state in Fourier space, i.e., `self.dt` time units later.
        """
        pass
