from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Complex, Float

from ._spectral import (
    build_derivative_operator,
    fft,
    ifft,
    spatial_shape,
    wavenumber_shape,
)
from .etdrk import ETDRK0, ETDRK1, ETDRK2, ETDRK3, ETDRK4, BaseETDRK
from .nonlin_fun import BaseNonlinearFun


class BaseStepper(eqx.Module, ABC):
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
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Baseclass for timesteppers based on Fourier pseudo-spectral Exponential
        Time Differencing Runge Kutta methods (ETDRK); efficiently solving
        semi-linear PDEs of the form

            uₜ = ℒu + 𝒩(u)

        with a linear differential operator ℒ and a nonlinear differential
        operator 𝒩(...).

        A subclass must implement the methods `_build_linear_operator` and
        `_build_nonlinear_fun`. The former returns the diagonal linear operator
        in Fourier space. The latter returns a subclass of `BaseNonlinearFun`.
        See the `exponax.ic` submodule for pre-defined nonlinear operators and
        how to subclass your own.

        Save attributes specific to the concrete PDE before calling the parent
        constructor because it will call the abstract methods.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `num_channels`: The number of channels `C` in the state vector/tensor.
            For most problem, like simple linear PDEs this will be one (because
            the temperature field in a heat/diffusion PDE is a scalar field).
            Some other problems like Burgers equation in higher dimensions or
            reaction-diffusion equations with multiple species will have more
            than one channel. This information is only used to check the shape
            of the input state vector in the `__call__` method. (keyword-only)
        - `order`: The order of the ETDRK method to use. Must be one of {0, 1,
            2, 3, 4}. The option `0` only solves the linear part of the
            equation. Hence, only use this for linear PDEs. For nonlinear PDEs,
            a higher order method tends to be more stable and accurate. `2` is
            often a good compromis in single-precision. Use `4` together with
            double precision (`jax.config.update("jax_enable_x64", True)`) for
            highest accuracy. (keyword-only)
        - `num_circle_points`: The number of points to use on the unit circle
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.
        """
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
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 2:
            self._integrator = ETDRK2(
                dt,
                linear_operator,
                nonlinear_fun,
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 3:
            self._integrator = ETDRK3(
                dt,
                linear_operator,
                nonlinear_fun,
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 4:
            self._integrator = ETDRK4(
                dt,
                linear_operator,
                nonlinear_fun,
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        else:
            raise NotImplementedError(f"Order {order} not implemented.")

    @abstractmethod
    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Assemble the L operator in Fourier space.

        **Arguments:**

        - `derivative_operator`: The derivative operator, shape `( D, ...,
            N//2+1 )`. The ellipsis are (D-1) axis of size N (**not** of size
            N//2+1).

        **Returns:**

        - `L`: The linear operator, shape `( C, ..., N//2+1 )`.
        """
        pass

    @abstractmethod
    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> BaseNonlinearFun:
        """
        Build the function that evaluates nonlinearity in physical space,
        transforms to Fourier space, and evaluates derivatives there.

        **Arguments:**

        - `derivative_operator`: The derivative operator, shape `( D, ...,
            N//2+1 )`.

        **Returns:**

        - `nonlinear_fun`: A function that evaluates the nonlinearities in
            time space, transforms to Fourier space, and evaluates the
            derivatives there. Should be a subclass of `BaseNonlinearFun`.
        """
        pass

    def step(self, u: Float[Array, "C ... N"]) -> Float[Array, "C ... N"]:
        """
        Perform one step of the time integration.

        **Arguments:**

        - `u`: The state vector, shape `(C, ..., N,)`.

        **Returns:**

        - `u_next`: The state vector after one step, shape `(C, ..., N,)`.
        """
        u_hat = fft(u, num_spatial_dims=self.num_spatial_dims)
        u_next_hat = self.step_fourier(u_hat)
        u_next = ifft(
            u_next_hat,
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
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
        Perform one step of the time integration for a single state.

        **Arguments:**

        - `u`: The state vector, shape `(C, ..., N,)`.

        **Returns:**

        - `u_next`: The state vector after one step, shape `(C, ..., N,)`.

        !!! tip
            Use this call method together with `exponax.rollout` to efficiently
            produce temporal trajectories.

        !!! info
            For batched operation, use `jax.vmap` on this function.
        """
        expected_shape = (self.num_channels,) + spatial_shape(
            self.num_spatial_dims, self.num_points
        )
        if u.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {u.shape}. For batched operation use `jax.vmap` on this function."
            )
        return self.step(u)
