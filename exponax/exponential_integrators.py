import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Complex, Array, Float
from typing import Callable

from .nonlinear_functions import BaseNonlinearFun, ZeroNonlinearFun

# E can either be 1 (single channel) or num_channels (multi-channel) for either
# the same linear operator for each channel or a different linear operator for
# each channel, respectively.
#
# So far, we do **not** support channel mixing via the linear operator (for
# example if we solved the wave equation or the sine-Gordon equation).


class BaseETDRK(eqx.Module):
    dt: float
    _exp_term: Complex[Array, "E ... (N//2)+1"]

    def __init__(
        self,
        dt: float,
        linear_operator: Complex[Array, "E ... (N//2)+1"],
    ):
        self.dt = dt
        self._exp_term = jnp.exp(self.dt * linear_operator)

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Advance the state in Fourier space.
        """
        raise NotImplementedError("Must be implemented by subclass")


class ETDRK0(BaseETDRK):
    """
    Exactly solve a linear PDE in Fourier space
    """

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return self._exp_term * u_hat


def roots_of_unity(M: int) -> Complex[Array, "M"]:
    """
    Return (complex-valued) array with M roots of unity.
    """
    # return jnp.exp(1j * jnp.pi * (jnp.arange(1, M+1) - 0.5) / M)
    return jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)


class ETDRK1(BaseETDRK):
    _nonlinear_fun: BaseNonlinearFun
    _coef_1: Complex[Array, "E ... (N//2)+1"]

    def __init__(
        self,
        dt: float,
        linear_operator: Complex[Array, "E ... (N//2)+1"],
        nonlinear_fun: BaseNonlinearFun,
        *,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        super().__init__(dt, linear_operator)
        self._nonlinear_fun = nonlinear_fun

        LR = (
            circle_radius * roots_of_unity(n_circle_points)
            + linear_operator[..., jnp.newaxis] * dt
        )

        self._coef_1 = dt * jnp.mean((jnp.exp(LR) - 1) / LR, axis=-1).real

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return self._exp_term * u_hat + self._coef_1 * self._nonlinear_fun(u_hat)


class ETDRK2(BaseETDRK):
    _nonlinear_fun: BaseNonlinearFun
    _coef_1: Complex[Array, "E ... (N//2)+1"]
    _coef_2: Complex[Array, "E ... (N//2)+1"]

    def __init__(
        self,
        dt: float,
        linear_operator: Complex[Array, "E ... (N//2)+1"],
        nonlinear_fun: BaseNonlinearFun,
        *,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        super().__init__(dt, linear_operator)
        self._nonlinear_fun = nonlinear_fun

        LR = (
            circle_radius * roots_of_unity(n_circle_points)
            + linear_operator[..., jnp.newaxis] * dt
        )

        self._coef_1 = dt * jnp.mean((jnp.exp(LR) - 1) / LR, axis=-1).real

        self._coef_2 = dt * jnp.mean((jnp.exp(LR) - 1 - LR) / LR**2, axis=-1).real

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_nonlin_hat = self._nonlinear_fun(u_hat)
        u_stage_1_hat = self._exp_term * u_hat + self._coef_1 * u_nonlin_hat

        u_stage_1_nonlin_hat = self._nonlinear_fun(u_stage_1_hat)
        u_next_hat = u_stage_1_hat + self._coef_2 * (
            u_stage_1_nonlin_hat - u_nonlin_hat
        )
        return u_next_hat


class ETDRK3(BaseETDRK):
    _nonlinear_fun: BaseNonlinearFun
    _half_exp_term: Complex[Array, "E ... (N//2)+1"]
    _coef_1: Complex[Array, "E ... (N//2)+1"]
    _coef_2: Complex[Array, "E ... (N//2)+1"]
    _coef_3: Complex[Array, "E ... (N//2)+1"]
    _coef_4: Complex[Array, "E ... (N//2)+1"]
    _coef_5: Complex[Array, "E ... (N//2)+1"]

    def __init__(
        self,
        dt: float,
        linear_operator: Complex[Array, "E ... (N//2)+1"],
        nonlinear_fun: BaseNonlinearFun,
        *,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        super().__init__(dt, linear_operator)
        self._nonlinear_fun = nonlinear_fun
        self._half_exp_term = jnp.exp(0.5 * dt * linear_operator)

        LR = (
            circle_radius * roots_of_unity(n_circle_points)
            + linear_operator[..., jnp.newaxis] * dt
        )

        self._coef_1 = dt * jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=-1).real

        self._coef_2 = dt * jnp.mean((jnp.exp(LR) - 1) / LR, axis=-1).real

        self._coef_3 = (
            dt
            * jnp.mean(
                (-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / (LR**3), axis=-1
            ).real
        )

        self._coef_4 = (
            dt
            * jnp.mean(
                (4.0 * (2.0 + LR + jnp.exp(LR) * (-2 + LR))) / (LR**3), axis=-1
            ).real
        )

        self._coef_5 = (
            dt
            * jnp.mean(
                (-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / (LR**3), axis=-1
            ).real
        )

    def step_fourier(
        self,
        u_hat: Complex[Array, "E ... (N//2)+1"],
    ) -> Complex[Array, "E ... (N//2)+1"]:
        u_nonlin_hat = self._nonlinear_fun(u_hat)
        u_stage_1_hat = self._half_exp_term * u_hat + self._coef_1 * u_nonlin_hat

        u_stage_1_nonlin_hat = self._nonlinear_fun(u_stage_1_hat)
        u_stage_2_hat = self._exp_term * u_hat + self._coef_2 * (
            2 * u_stage_1_nonlin_hat - u_nonlin_hat
        )

        u_stage_2_nonlin_hat = self._nonlinear_fun(u_stage_2_hat)

        u_next_hat = (
            self._exp_term * u_hat
            + self._coef_3 * u_nonlin_hat
            + self._coef_4 * u_stage_1_nonlin_hat
            + self._coef_5 * u_stage_2_nonlin_hat
        )

        return u_next_hat


class ETDRK4(BaseETDRK):
    _nonlinear_fun: BaseNonlinearFun
    _half_exp_term: Complex[Array, "E ... (N//2)+1"]
    _coef_1: Complex[Array, "E ... (N//2)+1"]
    _coef_2: Complex[Array, "E ... (N//2)+1"]
    _coef_3: Complex[Array, "E ... (N//2)+1"]
    _coef_4: Complex[Array, "E ... (N//2)+1"]
    _coef_5: Complex[Array, "E ... (N//2)+1"]
    _coef_6: Complex[Array, "E ... (N//2)+1"]

    def __init__(
        self,
        dt: float,
        linear_operator: Complex[Array, "E ... (N//2)+1"],
        nonlinear_fun: BaseNonlinearFun,
        *,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        super().__init__(dt, linear_operator)
        self._nonlinear_fun = nonlinear_fun
        self._half_exp_term = jnp.exp(0.5 * dt * linear_operator)

        LR = (
            circle_radius * roots_of_unity(n_circle_points)
            + linear_operator[..., jnp.newaxis] * dt
        )

        self._coef_1 = dt * jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=-1).real

        self._coef_2 = self._coef_1
        self._coef_3 = self._coef_1

        self._coef_4 = (
            dt
            * jnp.mean(
                (-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / (LR**3), axis=-1
            ).real
        )

        self._coef_5 = (
            dt * jnp.mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / (LR**3), axis=-1).real
        )

        self._coef_6 = (
            dt
            * jnp.mean(
                (-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / (LR**3), axis=-1
            ).real
        )

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_nonlin_hat = self._nonlinear_fun(u_hat)
        u_stage_1_hat = self._half_exp_term * u_hat + self._coef_1 * u_nonlin_hat

        u_stage_1_nonlin_hat = self._nonlinear_fun(u_stage_1_hat)
        u_stage_2_hat = (
            self._half_exp_term * u_hat + self._coef_2 * u_stage_1_nonlin_hat
        )

        u_stage_2_nonlin_hat = self._nonlinear_fun(u_stage_2_hat)
        u_stage_3_hat = self._half_exp_term * u_stage_1_hat + self._coef_3 * (
            2 * u_stage_2_nonlin_hat - u_nonlin_hat
        )

        u_stage_3_nonlin_hat = self._nonlinear_fun(u_stage_3_hat)

        u_next_hat = (
            self._exp_term * u_hat
            + self._coef_4 * u_nonlin_hat
            + self._coef_5 * 2 * (u_stage_1_nonlin_hat + u_stage_2_nonlin_hat)
            + self._coef_6 * u_stage_3_nonlin_hat
        )

        return u_next_hat
