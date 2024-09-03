import equinox as eqx
from jaxtyping import Array, Complex, Float

from ._base_stepper import BaseStepper
from ._spectral import fft, ifft, spatial_shape
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

        !!! info
            Performs the substepping in Fourier space to avoid unnecessary
            back-and-forth transformations.

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
        Step the PDE forward in time by `self.num_sub_steps` time steps given the
        current state `u`.

        !!! info
            Performs the substepping in Fourier space to avoid unnecessary
            back-and-forth transformations.

        **Arguments:**

        - `u`: The current state.

        **Returns:**

        - `u_next`: The state after `self.num_sub_steps` time steps.
        """
        u_hat = fft(u, num_spatial_dims=self.num_spatial_dims)
        u_hat_after_steps = self.step_fourier(u_hat)
        u_after_steps = ifft(
            u_hat_after_steps,
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
        )
        return u_after_steps

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Step the PDE forward in time by self.num_sub_steps time steps given the
        current state `u_hat` in real-valued Fourier space.

        **Arguments:**

        - `u_hat`: The current state in Fourier space.

        **Returns:**

        - `u_next_hat`: The state after `self.num_sub_steps` time steps in Fourier
            space.
        """
        return repeat(self.stepper.step_fourier, self.num_sub_steps)(u_hat)

    def __call__(
        self,
        u: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Step the PDE forward in time by self.num_sub_steps time steps given the
        current state `u`.

        !!! info
            Performs the substepping in Fourier space to avoid unnecessary
            back-and-forth transformations.

        **Arguments:**

        - `u`: The current state.

        **Returns:**

        - `u_next`: The state after `self.num_sub_steps` time steps.

        !!! tip
            Use this call method together with `exponax.rollout` to efficiently
            produce temporal trajectories.

        !!! info
            For batched operation, use `jax.vmap` on this function.
        """
        expected_shape = (self.num_channels,) + spatial_shape(
            self.num_spatial_dims, self.num_points
        )
        if u.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {u.shape}. For batched operation use `jax.vmap` on this function."
            )
        return self.step(u)
