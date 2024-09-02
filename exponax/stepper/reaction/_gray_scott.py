import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ..._spectral import build_laplace_operator
from ...nonlin_fun import BaseNonlinearFun


class GrayScottNonlinearFun(BaseNonlinearFun):
    feed_rate: float
    kill_rate: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: float,
        feed_rate: float,
        kill_rate: float,
    ):
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        num_channels = u_hat.shape[0]
        if num_channels != 2:
            raise ValueError("num_channels must be 2")
        u = self.ifft(self.dealias(u_hat))
        u_power = jnp.stack(
            [
                self.feed_rate * (1 - u[0]) - u[0] * u[1] ** 2,
                -(self.feed_rate + self.kill_rate) * u[1] + u[0] * u[1] ** 2,
            ]
        )
        u_power_hat = self.fft(u_power)
        return u_power_hat


class GrayScott(BaseStepper):
    diffusivity_1: float
    diffusivity_2: float
    feed_rate: float
    kill_rate: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity_1: float = 2e-5,
        diffusivity_2: float = 1e-5,
        feed_rate: float = 0.04,
        kill_rate: float = 0.06,
        order: int = 2,
        # Needs lower value due to cubic nonlinearity
        dealiasing_fraction: float = 1 / 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Gray-Scott reaction
        diffusion equation on periodic boundary conditions. This
        reaction-diffusion models the interaction of two chemical species u & v.

        In 1d, the Gray-Scott equation is given by

        ```
            uₜ = ν₁ uₓₓ + f(1 - u) - u v²

            vₜ = ν₂ vₓₓ - (f + k) v + u v²
        ```

        with `ν₁` and `ν₂` the diffusivities, `f` the feed rate, and `k` the
        kill rate. No matter the spatial dimension, this dynamics always has two
        channels, refering to the two chemical species. In higher dimensions,
        the equations read

        ```
            uₜ = ν₁ Δu + f(1 - u) - u v²

            vₜ = ν₂ Δv - (f + k) v + u v²
        ```

        with `Δ` the Laplacian.

        The Gray-Scott equation is known to produce a variety of patterns, such
        as spots, stripes, and spirals. The expected temporal behavior is highly
        dependent on the values of the feed and kill rates, see also this paper:
        https://www.ljll.fr/hecht/ftp/ff++/2015-cimpa-IIT/edp-tuto/Pearson.pdf

        IMPORTANT: Both channels are expected to have values in the range `[0,
        1]`.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `diffusivity_1`: The diffusivity `ν₁` of the first species.
            Default is `2e-5`.
        - `diffusivity_2`: The diffusivity `ν₂` of the second species.
            Default is `1e-5`.
        - `feed_rate`: The feed rate `f`. Default is `0.04`.
        - `kill_rate`: The kill rate `k`. Default is `0.06`.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for higher
            accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. Default: 1/2.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.

        TODO: Translate the different configurations of
        https://www.ljll.fr/hecht/ftp/ff++/2015-cimpa-IIT/edp-tuto/Pearson.pdf
        """
        self.diffusivity_1 = diffusivity_1
        self.diffusivity_2 = diffusivity_2
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=2,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "2 ... (N//2)+1"]:
        laplace = build_laplace_operator(derivative_operator, order=2)
        linear_operator = jnp.concatenate(
            [
                self.diffusivity_1 * laplace,
                self.diffusivity_2 * laplace,
            ]
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> GrayScottNonlinearFun:
        return GrayScottNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            feed_rate=self.feed_rate,
            kill_rate=self.kill_rate,
            dealiasing_fraction=self.dealiasing_fraction,
        )
