import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..nonlin_fun import BaseNonlinearFun
from ._base_etdrk import BaseETDRK
from ._utils import roots_of_unity


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
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        r"""
        Solve a semi-linear PDE using Exponential Time Differencing Runge-Kutta
        with a **third order approximation**.

        Adapted from Eq. (23-25) of [Cox and Matthews
        (2002)](https://doi.org/10.1006/jcph.2002.6995):

        $$
            \begin{aligned}
                \hat{u}_h^* &= \exp(\hat{\mathcal{L}}_h \Delta t / 2) \odot \hat{u}_h^{[t]} + \frac{\exp(\hat{\mathcal{L}}_h \Delta t/2) - 1}{\hat{\mathcal{L}}_h} \odot \hat{\mathcal{N}}_h(\hat{u}_h^{[t]}).
                \\
                \hat{u}_h^{**} &= \exp(\hat{\mathcal{L}}_h \Delta t / 2) \odot \hat{u}_h^{[t]} + \frac{\exp(\hat{\mathcal{L}}_h \Delta t) - 1}{\hat{\mathcal{L}}_h} \odot \left( 2 \hat{\mathcal{N}}_h(\hat{u}_h^*) - \hat{\mathcal{N}}_h(\hat{u}_h^{[t]}) \right).
                \\
                \hat{u}_h^{[t+1]} &= \exp(\hat{\mathcal{L}}_h \Delta t) \odot \hat{u}_h^{[t]}
                \\
                &+ \frac{-4 - \exp(\hat{\mathcal{L}}_h \Delta t) + \exp(\hat{\mathcal{L}}_h \Delta) \left( 4 - 3 \hat{\mathcal{L}}_h \Delta t + \left(\hat{\mathcal{L}}_h \Delta t\right)^2 \right)}{\hat{\mathcal{L}}_h^3 (\Delta t)^2} \odot \hat{\mathcal{N}}_h(\hat{u}_h^{[t]}).
                \\
                &+ 4 \frac{2 + \hat{\mathcal{L}}_h \Delta t + \exp(\hat{\mathcal{L}}_h \Delta t) \left( -2 + \hat{\mathcal{L}}_h \Delta t \right)}{\hat{\mathcal{L}}_h^3 (\Delta t)^2} \odot \hat{\mathcal{N}}_h(\hat{u}_h^*)
                \\
                &+ \frac{-4 - 3 \hat{\mathcal{L}}_h \Delta t - \left( \hat{\mathcal{L}}_h \Delta t \right)^2 + \exp(\hat{\mathcal{L}}_h \Delta t) \left( 4 - \hat{\mathcal{L}}_h \Delta t \right)}{\hat{\mathcal{L}}_h^3 (\Delta t)^2} \odot \hat{\mathcal{N}}_h(\hat{u}_h^{**})
            \end{aligned}
        $$

        where $\hat{\mathcal{N}}_h$ is the Fourier pseudo-spectral treatment of
        the nonlinear differential operator.

        **Arguments:**

        - `dt`: The time step size.
        - `linear_operator`: The linear operator of the PDE. Must have a leading
            channel axis, followed by one, two or three spatial axes whereas the
            last axis must be of size `(N//2)+1` where `N` is the number of
            dimensions in the former spatial axes.
        - `nonlinear_fun`: The Fourier pseudo-spectral treatment of the
            nonlinear differential operator. ! The operator must take care of
            proper dealiasing.
        - `num_circle_points`: The number of points on the unit circle used to
            approximate the numerically challenging coefficients.
        - `circle_radius`: The radius of the circle used to approximate the
            numerically challenging coefficients.

        !!! warning
            The nonlinear function must take care of proper dealiasing.

        !!! note
            The numerically stable evaluation of the coefficients follows
            [Kassam and Trefethen
            (2005)](https://doi.org/10.1137/S1064827502410633).
        """
        super().__init__(dt, linear_operator)
        self._nonlinear_fun = nonlinear_fun
        self._half_exp_term = jnp.exp(0.5 * dt * linear_operator)

        LR = (
            circle_radius * roots_of_unity(num_circle_points)
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
