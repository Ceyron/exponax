from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class PolynomialNonlinearFun(BaseNonlinearFun):
    """
    Channel-separate evaluation; and no mixed terms.
    """

    coefficients: tuple[float, ...]  # Starting from order 0

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: float,
        coefficients: tuple[float, ...],
    ):
        """
        Coefficient list starts from order 0.
        """
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.coefficients = coefficients

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u = self.ifft(self.dealias(u_hat))
        u_power = 1.0
        u_nonlin = 0.0
        for coeff in self.coefficients:
            u_nonlin += coeff * u_power
            u_power = u_power * u

        u_nonlin_hat = self.fft(u_nonlin)
        return u_nonlin_hat
