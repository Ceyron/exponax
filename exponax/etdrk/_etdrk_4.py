import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Inexact

from ..nonlin_fun import BaseNonlinearFun
from ._base_etdrk import BaseETDRK
from ._utils import roots_of_unity


class ETDRK4(BaseETDRK):
    _nonlinear_fun: BaseNonlinearFun
    _half_exp_term: Complex[Array, "E ... (N//2)+1"]
    _coef_1: Inexact[Array, "E ... (N//2)+1"]
    _coef_2: Inexact[Array, "E ... (N//2)+1"]
    _coef_3: Inexact[Array, "E ... (N//2)+1"]
    _coef_4: Inexact[Array, "E ... (N//2)+1"]
    _coef_5: Inexact[Array, "E ... (N//2)+1"]
    _coef_6: Inexact[Array, "E ... (N//2)+1"]

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
        with a **fourth order approximation**.

        Adapted from Eq. (26-29) of [Cox and Matthews
        (2002)](https://doi.org/10.1006/jcph.2002.6995):

        $$
        \begin{aligned}
            \hat{u}_h^* &=
            \exp(\hat{\mathcal{L}}_h \Delta t / 2)
            \odot
            \hat{u}_h^{[t]}
            +
            \frac{
                \exp(\hat{\mathcal{L}}_h \Delta t/2) - 1
            }{
                \hat{\mathcal{L}}_h
            }
            \odot
            \hat{\mathcal{N}}_h(\hat{u}_h^{[t]}).
            \\
            \hat{u}_h^{**}
            &=
            \exp(\hat{\mathcal{L}}_h \Delta t / 2)
            \odot
            \hat{u}_h^{[t]}
            +
            \frac{
                \exp(\hat{\mathcal{L}}_h \Delta t / 2) - 1
            }{
                \hat{\mathcal{L}}_h
            } \odot \hat{\mathcal{N}}_h(\hat{u}_h^*).
            \\
            \hat{u}_h^{***}
            &=
            \exp(\hat{\mathcal{L}}_h \Delta t)
            \odot
            \hat{u}_h^{*}
            +
            \frac{
                \exp(\hat{\mathcal{L}}_h \Delta t/2) - 1
            }{
                \hat{\mathcal{L}}_h
            }
            \odot
            \left(
                2 \hat{\mathcal{N}}_h(\hat{u}_h^{**})
                -
                \hat{\mathcal{N}}_h(\hat{u}_h^{[t]})
            \right).
            \\
            \hat{u}_h^{[t+1]}
            &=
            \exp(\hat{\mathcal{L}}_h \Delta t)
            \odot
            \hat{u}_h^{[t]}
            \\
            &+
            \frac{
                -4 - \hat{\mathcal{L}}_h \Delta t
                +
                \exp(\hat{\mathcal{L}}_h \Delta t)
                \left(
                    4 - 3 \hat{\mathcal{L}}_h \Delta t
                    +
                    \left(
                        \hat{\mathcal{L}}_h \Delta t
                    \right)^2
                \right)
            }{
                \hat{\mathcal{L}}_h^3 (\Delta t)^2
            }
            \odot
            \hat{\mathcal{N}}_h(\hat{u}_h^{[t]})
            \\
            &+
            2 \frac{
                2 + \hat{\mathcal{L}}_h \Delta t
                +
                \exp(\hat{\mathcal{L}}_h \Delta t)
                \left(
                    -2 + \hat{\mathcal{L}}_h \Delta t
                \right)
            }{
                \hat{\mathcal{L}}_h^3 (\Delta t)^2
            }
            \odot
            \left(
                \hat{\mathcal{N}}_h(\hat{u}_h^*)
                +
                \hat{\mathcal{N}}_h(\hat{u}_h^{**})
            \right)
            \\
            &+
            \frac{
                -4 - 3 \hat{\mathcal{L}}_h \Delta t
                - \left(
                    \hat{\mathcal{L}}_h \Delta t
                \right)^2
                + \exp(\hat{\mathcal{L}}_h \Delta t)
                \left(
                    4 - \hat{\mathcal{L}}_h \Delta t
                \right)
            }{
                \hat{\mathcal{L}}_h^3 (\Delta t)^2
            }
            \odot
            \hat{\mathcal{N}}_h(\hat{u}_h^{***})
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
            nonlinear differential operator.
        - `num_circle_points`: The number of points on the unit circle used to
            approximate the numerically challenging coefficients.
        - `circle_radius`: The radius of the circle used to approximate the
            numerically challenging coefficients.

        !!! warning
            The nonlinear function must take care of proper dealiasing.
            `BaseNonlinearFun` handles this automatically via its `fft` and
            `ifft` methods which apply pre- and post-dealiasing.

        !!! note
            The numerically stable evaluation of the coefficients follows
            [Kassam and Trefethen
            (2005)](https://doi.org/10.1137/S1064827502410633).

        !!! note
            If the linear operator has non-zero imaginary part (e.g., for
            advection or dispersion), the precomputed coefficients are stored
            as complex-valued arrays, doubling the memory compared to a purely
            real operator (e.g., pure diffusion).
        """
        super().__init__(dt, linear_operator)
        self._nonlinear_fun = nonlinear_fun
        self._half_exp_term = jnp.exp(0.5 * dt * linear_operator)

        roots = roots_of_unity(num_circle_points)
        L_dt = linear_operator * dt
        is_real_operator = bool(jnp.all(linear_operator.imag == 0))

        if is_real_operator:

            def scan_body(accs, root):
                lr = circle_radius * root + L_dt
                exp_lr = jnp.exp(lr)
                exp_lr_half = jnp.exp(lr / 2)
                c1 = ((exp_lr_half - 1) / lr).real
                c4 = ((-4 - lr + exp_lr * (4 - 3 * lr + lr**2)) / lr**3).real
                c5 = ((2 + lr + exp_lr * (-2 + lr)) / lr**3).real
                c6 = ((-4 - 3 * lr - lr**2 + exp_lr * (4 - lr)) / lr**3).real
                return (
                    accs[0] + c1,
                    accs[1] + c4,
                    accs[2] + c5,
                    accs[3] + c6,
                ), None

            zeros = jnp.zeros_like(L_dt.real)
        else:

            def scan_body(accs, root):
                lr = circle_radius * root + L_dt
                exp_lr = jnp.exp(lr)
                exp_lr_half = jnp.exp(lr / 2)
                c1 = (exp_lr_half - 1) / lr
                c4 = (-4 - lr + exp_lr * (4 - 3 * lr + lr**2)) / lr**3
                c5 = (2 + lr + exp_lr * (-2 + lr)) / lr**3
                c6 = (-4 - 3 * lr - lr**2 + exp_lr * (4 - lr)) / lr**3
                return (
                    accs[0] + c1,
                    accs[1] + c4,
                    accs[2] + c5,
                    accs[3] + c6,
                ), None

            zeros = jnp.zeros_like(L_dt)

        (s1, s4, s5, s6), _ = jax.lax.scan(scan_body, (zeros,) * 4, roots)
        mean_c1 = s1 / num_circle_points
        mean_c4 = s4 / num_circle_points
        mean_c5 = s5 / num_circle_points
        mean_c6 = s6 / num_circle_points
        self._coef_1 = dt * mean_c1
        self._coef_2 = self._coef_1
        self._coef_3 = self._coef_1
        self._coef_4 = dt * mean_c4
        self._coef_5 = dt * mean_c5
        self._coef_6 = dt * mean_c6

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
