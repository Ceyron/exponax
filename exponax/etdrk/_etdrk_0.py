from jaxtyping import Array, Complex

from ._base_etdrk import BaseETDRK


class ETDRK0(BaseETDRK):
    def __init__(
        self,
        dt: float,
        linear_operator: Complex[Array, "E ... (N//2)+1"],
    ):
        r"""
        Exactly solve a linear PDE in Fourier space.

        $$
            \hat{u}_h^{[t+1]} = \exp(\hat{\mathcal{L}}_h \Delta t) \odot
            \hat{u}_h^{[t]}
        $$

        **Arguments:**

        - `dt`: The time step size.
        - `linear_operator`: The linear operator of the PDE. Must have a leading
            channel axis, followed by one, two or three spatial axes whereas the
            last axis must be of size `(N//2)+1` where `N` is the number of
            dimensions in the former spatial axes.
        """
        super().__init__(dt, linear_operator)

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return self._exp_term * u_hat
