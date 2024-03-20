from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class PolynomialNonlinearFun(BaseNonlinearFun):
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
        Performs a pseudo-spectral evaluation of an (unmixed) polynomial
        nonlineariy, e.g., as they are found in reaction-diffusion systems
        (e.g., FisherKPP). In state space, this reads

        ```
            𝒩(u) = ∑ₖ cₖ uᵏ
        ```

        with `cₖ` the coefficient of the `k`-th order term. Note that there is
        no channe mixing. For example, `u₀² u₁` cannot be represented.

        This format works in any number of dimensions.

        **Arguments:**
            - `num_spatial_dims`: The number of spatial dimensions `d`.
            - `num_points`: The number of points `N` used to discretize the
                domain. This **includes** the left boundary point and
                **excludes** the right boundary point. In higher dimensions; the
                number of points in each dimension is the same.
            - `dealiasing_fraction`: The fraction of the highest resolved modes
                that are not aliased. Defaults to `2/3` which corresponds to
                Orszag's 2/3 rule which is sufficient for up to quadratic
                nonlinearities. Higher order nonlinearities might require a
                higher dealiasing fraction.
            - `coefficients`: The coefficients of the polynomial terms. The
                coefficients are expected to be given in the order of increasing
                order. The list starts with the zeroth order. For example for a
                purely quadratic nonlinearity, the coefficients are `(0.0, 0.0,
                1.0)`.

        **Notes:**
            - A zeroth-order term is independent of `u` and essentially acts as
                a constant forcing term.
            - A first-order term is linear and could also be represented in the
                linear part of the timestepper where it would be represented
                analytically.
            - As such it is often the case, that the coefficients tuple starts
                with two zeros.
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
