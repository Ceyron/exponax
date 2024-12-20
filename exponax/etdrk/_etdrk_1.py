import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..nonlin_fun import BaseNonlinearFun
from ._base_etdrk import BaseETDRK
from ._utils import roots_of_unity


class ETDRK1(BaseETDRK):
    _nonlinear_fun: BaseNonlinearFun
    _coef_1: Complex[Array, "E ... (N//2)+1"]

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
        with a **first order approximation**.

        Adapted from Eq. (4) of [Cox and Matthews
        (2002)](https://doi.org/10.1006/jcph.2002.6995):

        $$
            \hat{u}_h^{[t+1]} = \exp(\hat{\mathcal{L}}_h \Delta t) \odot
            \hat{u}_h^{[t]} + \frac{\exp(\hat{\mathcal{L}}_h \Delta t) -
            1}{\hat{\mathcal{L}}_h} \odot \hat{\mathcal{N}}_h(\hat{u}_h^{[t]})
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

        !!! note
            The numerically stable evaluation of the coefficients follows
            [Kassam and Trefethen
            (2005)](https://doi.org/10.1137/S1064827502410633).
        """
        super().__init__(dt, linear_operator)
        self._nonlinear_fun = nonlinear_fun

        LR = (
            circle_radius * roots_of_unity(num_circle_points)
            + linear_operator[..., jnp.newaxis] * dt
        )

        self._coef_1 = dt * jnp.mean((jnp.exp(LR) - 1) / LR, axis=-1).real

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return self._exp_term * u_hat + self._coef_1 * self._nonlinear_fun(u_hat)
