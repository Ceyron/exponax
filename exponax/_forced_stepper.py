import equinox as eqx
from jaxtyping import Array, Complex, Float

from ._base_stepper import BaseStepper


class ForcedStepper(eqx.Module):
    stepper: BaseStepper

    def __init__(
        self,
        stepper: BaseStepper,
    ):
        """
        Transform a stepper of signature `(u,) -> u_next` into a stepper of
        signature `(u, f) -> u_next` that also accepts a forcing vector `f`.

        Transforms a stepper for a PDE of the form u_t = Lu + N(u) into a stepper
        for a PDE of the form u_t = Lu + N(u) + f, where f is a forcing term. For
        this, we split by operators

            v_t = f

            u_t = Lv + N(v)

        Since we assume to only have access to the forcing function evaluated at one
        time level (but on the same grid as the state), we use a forward Euler
        scheme to integrate the first equation. The second equation is integrated
        using the original stepper.

        Note: This operator splitting makes the total scheme only first order
        accurate in time. It is a quick hack to extend the other sophisticated
        transient integrators to forced problems.

        **Arguments**:

        - `stepper`: The stepper to be transformed.
        """
        self.stepper = stepper

    def step(
        self,
        u: Float[Array, "C ... N"],
        f: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Step the PDE forward in time by one time step given the current state
        `u` and the forcing term `f`.

        The forcing term `f` is assumed to be evaluated on the same grid as `u`.

        **Arguments**:

        - `u`: The current state.
        - `f`: The forcing term.

        **Returns**:

        - `u_next`: The state after one time step.
        """
        u_with_force = u + self.stepper.dt * f
        return self.stepper.step(u_with_force)

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
        f_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Step the PDE forward in time by one time step given the current state
        `u_hat` in Fourier space and the forcing term `f_hat` in Fourier space.

        The forcing term `f_hat` is assumed to be evaluated on the same grid as
        `u_hat`.

        **Arguments**:

        - `u_hat`: The current state in Fourier space.
        - `f_hat`: The forcing term in Fourier space.

        **Returns**:

        - `u_next_hat`: The state after one time step in Fourier space.
        """
        u_hat_with_force = u_hat + self.stepper.dt * f_hat
        return self.stepper.step_fourier(u_hat_with_force)

    def __call__(
        self,
        u: Float[Array, "C ... N"],
        f: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Step the PDE forward in time by one time step given the current state
        `u` and the forcing term `f`.

        The forcing term `f` is assumed to be evaluated on the same grid as `u`.

        **Arguments:**

        - `u`: The current state.
        - `f`: The forcing term.

        **Returns:**

        - `u_next`: The state after one time step.
        """
        return self.step(u, f)
