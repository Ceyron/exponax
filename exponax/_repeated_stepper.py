import equinox as eqx
from jaxtyping import Array, Complex, Float

from ._base_stepper import BaseStepper
from ._utils import repeat


class RepeatedStepper(eqx.Module):
    num_spatial_dims: int
    domain_extent: float
    num_points: int
    num_channels: int
    dt: float
    dx: float

    stepper: BaseStepper
    num_sub_steps: int

    def __init__(
        self,
        stepper: BaseStepper,
        num_sub_steps: int,
    ):
        """
        Sugarcoat the utility function `repeat` in a callable PyTree for easy
        composition with other equinox modules.

        One intended usage is to get "more accurate" or "more stable" time steppers
        that perform substeps.

        The effective time step is `self.stepper.dt * self.num_sub_steps`. In order to
        get a time step of X with Y substeps, first instantiate a stepper with a
        time step of X/Y and then wrap it in a RepeatedStepper with num_sub_steps=Y.

        **Arguments:**
            - `stepper`: The stepper to repeat.
            - `num_sub_steps`: The number of substeps to perform.
        """
        self.stepper = stepper
        self.num_sub_steps = num_sub_steps

        self.dt = stepper.dt * num_sub_steps

        self.num_spatial_dims = stepper.num_spatial_dims
        self.domain_extent = stepper.domain_extent
        self.num_points = stepper.num_points
        self.num_channels = stepper.num_channels
        self.dx = stepper.dx

    def step(
        self,
        u: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Step the PDE forward in time by self.num_sub_steps time steps given the
        current state `u`.
        """
        return repeat(self.stepper.step, self.num_sub_steps)(u)

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Step the PDE forward in time by self.num_sub_steps time steps given the
        current state `u_hat` in real-valued Fourier space.
        """
        return repeat(self.stepper.step_fourier, self.num_sub_steps)(u_hat)

    def __call__(
        self,
        u: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Step the PDE forward in time by self.num_sub_steps time steps given the
        current state `u`.
        """
        return repeat(self.stepper, self.num_sub_steps)(u)
