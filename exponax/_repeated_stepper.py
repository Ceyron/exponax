import equinox as eqx
from jaxtyping import Array, Complex, Float

from ._base_stepper import BaseStepper
from ._utils import repeat


class RepeatedStepper(eqx.Module):
    """
    Sugarcoat the utility function `repeat` in a callable PyTree for easy
    composition with other equinox modules.

    One intended usage is to get "more accurate" or "more stable" time steppers
    that perform substeps.

    The effective time step is `self.stepper.dt * self.n_sub_steps`. In order to
    get a time step of X with Y substeps, first instantiate a stepper with a
    time step of X/Y and then wrap it in a RepeatedStepper with n_sub_steps=Y.

    **Arguments:**
        - `stepper`: The stepper to repeat.
        - `n_sub_steps`: The number of substeps to perform.
    """

    stepper: BaseStepper
    num_sub_steps: int

    def step(
        self,
        u: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Step the PDE forward in time by self.n_sub_steps time steps given the
        current state `u`.
        """
        return repeat(self.stepper.step, self.num_sub_steps)(u)

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Step the PDE forward in time by self.n_sub_steps time steps given the
        current state `u_hat` in real-valued Fourier space.
        """
        return repeat(self.stepper.step_fourier, self.num_sub_steps)(u_hat)

    def __call__(
        self,
        u: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Step the PDE forward in time by self.n_sub_steps time steps given the
        current state `u`.
        """
        return repeat(self.stepper, self.num_sub_steps)(u)
