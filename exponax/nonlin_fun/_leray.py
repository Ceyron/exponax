import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._spectral import build_laplace_operator
from ._base import BaseNonlinearFun


class Leray(BaseNonlinearFun):
    inv_laplacian: Complex[Array, " 1 ... (N//2+1) "]
    derivative_operator: Complex[Array, " D ... (N//2+1) "]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, " D ... (N//2+1) "],
        order: int = 2,
    ):
        """
        Technically not a nonlinear operator, but it performs channel mixing
        between the velocity channels.
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )

        laplace_operator = build_laplace_operator(derivative_operator, order=order)

        self.inv_laplacian = jnp.where(
            laplace_operator != 0, 1.0 / laplace_operator, 0.0
        )

        self.derivative_operator = derivative_operator

    def __call__(
        self,
        u_hat: Complex[Array, " D ... (N//2+1) "],
    ) -> Complex[Array, " D ... (N//2+1) "]:
        """Compute the Leray projection of the velocity field u_hat.

        Args:
            u_hat: Velocity field in Fourier space.

        Returns:
            The Leray projection of u_hat in Fourier space.
        """
        div_u_hat = jnp.sum(
            self.derivative_operator * u_hat,
            axis=0,
            keepdims=True,
        )

        pressure_hat = -self.inv_laplacian * div_u_hat

        grad_pressure_hat = self.derivative_operator * pressure_hat

        return u_hat + grad_pressure_hat
