import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._spectral import (
    build_derivative_operator,
    build_laplace_operator,
    fft,
    ifft,
    spatial_shape,
)


class Poisson(eqx.Module):
    num_spatial_dims: int
    domain_extent: float
    num_points: int
    dx: float

    _inv_operator: Complex[Array, "1 ... N"]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        *,
        order=2,
    ):
        """
        Exactly solve the Poisson equation with periodic boundary conditions.

        This "stepper" is different from all other steppers in this package in
        that it does not solve a time-dependent PDE. Instead, it solves the
        Poisson equation

        $$ u_{xx} = - f $$

        for a given right hand side $f$.

        It is included for completion.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions.
        - `domain_extent`: The extent of the domain.
        - `num_points`: The number of points in each spatial dimension.
        - `order`: The order of the Poisson equation. Defaults to 2. You can
            also set `order=4` for the biharmonic equation.
        """
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_points = num_points

        # Uses the convention that N does **not** include the right boundary
        # point
        self.dx = domain_extent / num_points

        derivative_operator = build_derivative_operator(
            num_spatial_dims, domain_extent, num_points
        )
        operator = build_laplace_operator(derivative_operator, order=order)

        # Uses mean zero solution
        self._inv_operator = jnp.where(operator == 0, 0.0, 1 / operator)

    def step_fourier(
        self,
        f_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Solve the Poisson equation in Fourier space.

        **Arguments:**

        - `f_hat`: The Fourier transform of the right hand side.

        **Returns:**

        - `u_hat`: The Fourier transform of the solution.
        """
        return -self._inv_operator * f_hat

    def step(
        self,
        f: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Solve the Poisson equation in state space.

        **Arguments:**

        - `f`: The right hand side.

        **Returns:**

        - `u`: The solution.
        """
        f_hat = fft(f, num_spatial_dims=self.num_spatial_dims)
        u_hat = self.step_fourier(f_hat)
        u = ifft(
            u_hat,
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
        )
        return u

    def __call__(
        self,
        f: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Solve the Poisson equation in state space.

        **Arguments:**

        - `f`: The right hand side.

        **Returns:**

        - `u`: The solution.
        """
        if f.shape[1:] != spatial_shape(self.num_spatial_dims, self.num_points):
            raise ValueError(
                f"Shape of f[1:] is {f.shape[1:]} but should be {spatial_shape(self.num_spatial_dims, self.num_points)}"
            )
        return self.step(f)
