from jaxtyping import Array, Complex

from ._base_etdrk import BaseETDRK


class ETDRK0(BaseETDRK):
    """
    Exactly solve a linear PDE in Fourier space
    """

    def step_fourier(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return self._exp_term * u_hat
