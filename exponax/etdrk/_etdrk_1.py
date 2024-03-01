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
