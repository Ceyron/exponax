import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._exponential_integrators import ETDRK0, ETDRK1, ETDRK2, ETDRK3, ETDRK4, BaseETDRK
from ._spectral import (
    build_derivative_operator,
    space_indices,
    spatial_shape,
    wavenumber_shape,
)
from .nonlin_fun import BaseNonlinearFun


class BaseStepper(eqx.Module):
    num_spatial_dims: int
    domain_extent: float
    num_points: int
    num_channels: int
    dt: float
    dx: float

    _integrator: BaseETDRK

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        num_channels: int,
        order: int,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_points = num_points
        self.dt = dt
        self.num_channels = num_channels

        # Uses the convention that N does **not** include the right boundary
        # point
        self.dx = domain_extent / num_points

        derivative_operator = build_derivative_operator(
            num_spatial_dims, domain_extent, num_points
        )

        linear_operator = self._build_linear_operator(derivative_operator)
        single_channel_shape = (1,) + wavenumber_shape(
            self.num_spatial_dims, self.num_points
        )  # Same operator for each channel (i.e., we broadcast)
        multi_channel_shape = (self.num_channels,) + wavenumber_shape(
            self.num_spatial_dims, self.num_points
        )  # Different operator for each channel
        if linear_operator.shape not in (single_channel_shape, multi_channel_shape):
            raise ValueError(
                f"Expected linear operator to have shape {single_channel_shape} or {multi_channel_shape}, got {linear_operator.shape}."
            )
        nonlinear_fun = self._build_nonlinear_fun(derivative_operator)

        if order == 0:
            self._integrator = ETDRK0(
                dt,
                linear_operator,
            )
        elif order == 1:
            self._integrator = ETDRK1(
                dt,
                linear_operator,
                nonlinear_fun,
                n_circle_points=n_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 2:
            self._integrator = ETDRK2(
                dt,
                linear_operator,
                nonlinear_fun,
                n_circle_points=n_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 3:
            self._integrator = ETDRK3(
                dt,
                linear_operator,
                nonlinear_fun,
                n_circle_points=n_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 4:
            self._integrator = ETDRK4(
                dt,
                linear_operator,
                nonlinear_fun,
                n_circle_points=n_circle_points,
                circle_radius=circle_radius,
            )
        else:
            raise NotImplementedError(f"Order {order} not implemented.")

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "D ... (N//2)+1"]:
        """
        Assemble the L operator in Fourier space.

        **Arguments:**
            - `derivative_operator`: The derivative operator, shape `( D, ...,
              N//2+1 )`. The ellipsis are (D-1) axis of size N (**not** of size
              N//2+1).

        **Returns:**
            - `L`: The linear operator, shape `( D, ..., N//2+1 )`.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> BaseNonlinearFun:
        """
        Build the function that evaluates nonlinearity in physical space,
        transforms to Fourier space, and evaluates derivatives there.

        **Arguments:**
            - `derivative_operator`: The derivative operator, shape `( D, ..., N//2+1 )`.

        **Returns:**
            - `nonlinear_fun`: A function that evaluates the nonlinearities in
                time space, transforms to Fourier space, and evaluates the
                derivatives there. Should be a subclass of `BaseNonlinearFun`.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def step(self, u: Float[Array, "C ... N"]) -> Float[Array, "C ... N"]:
        """
        Perform one step of the time integration.

        **Arguments:**
            - `u`: The state vector, shape `(C, ..., N,)`.

        **Returns:**
            - `u_next`: The state vector after one step, shape `(C, ..., N,)`.
        """
        u_hat = jnp.fft.rfftn(u, axes=space_indices(self.num_spatial_dims))
        u_next_hat = self.step_fourier(u_hat)
        u_next = jnp.fft.irfftn(
            u_next_hat,
            s=spatial_shape(self.num_spatial_dims, self.num_points),
            axes=space_indices(self.num_spatial_dims),
        )
        return u_next

    def step_fourier(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Perform one step of the time integration in Fourier space. Oftentimes,
        this is more efficient than `step` since it avoids back and forth
        transforms.

        **Arguments:**
            - `u_hat`: The (real) Fourier transform of the state vector

        **Returns:**
            - `u_next_hat`: The (real) Fourier transform of the state vector
                after one step
        """
        return self._integrator.step_fourier(u_hat)

    def __call__(
        self,
        u: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Performs a check
        """
        expected_shape = (self.num_channels,) + spatial_shape(
            self.num_spatial_dims, self.num_points
        )
        if u.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {u.shape}. For batched operation use `jax.vmap` on this function."
            )
        return self.step(u)
