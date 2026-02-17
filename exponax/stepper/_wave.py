import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from .._spectral import build_scaled_wavenumbers
from ..nonlin_fun import ZeroNonlinearFun


class Wave(BaseStepper):
    speed_of_sound: float
    wavenumber_norm: Float[Array, " 1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        speed_of_sound: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) wave equation on
        periodic boundary conditions.

        In 1d, the wave equation is given by

        ```
            uₜₜ = c² uₓₓ
        ```

        with `c ∈ ℝ` being the speed of sound (or wave speed).

        In higher dimensions, the wave equation is written using the Laplacian

        ```
            uₜₜ = c² Δu
        ```

        Internally, the second-order equation is rewritten as a first-order
        system of two coupled fields — height `h` and velocity `v = hₜ`:

        ```
            hₜ = v
            vₜ = c² Δh
        ```

        As a result, the state has **two channels**: `u[0]` is the height field
        `h` and `u[1]` is the velocity field `v`.

        **Diagonalization:**

        The general solution of the wave equation is a superposition of
        right-traveling and left-traveling waves (d'Alembert's decomposition).
        This stepper exploits that structure: rather than time-stepping the
        coupled `(h, v)` system directly, it transforms into independent
        traveling-wave modes that each evolve as a simple phase rotation.

        In Fourier space, each wavenumber `k` gives a 2×2 ODE for `(ĥ, v̂)`
        that oscillates at frequency `ω = c|k|` — analogous to a harmonic
        oscillator trading potential and kinetic energy. Three steps
        diagonalize it:

        1. **Rescale** — `h` and `v` live on different scales (displacement vs.
           rate). Defining `w = iωĥ` puts them on equal footing. The coupled
           system becomes symmetric: `wₜ = iω v̂`, `v̂ₜ = iω w`.

        2. **Rotate** — Taking the sum and difference
           `pos = (w + v̂)/√2`, `neg = (w − v̂)/√2` decouples the system
           into two independent modes: `posₜ = +iω · pos` and
           `negₜ = −iω · neg`. Physically, `pos` is the right-traveling
           wave and `neg` the left-traveling wave.

        3. **Exponentiate** — Each decoupled mode evolves as a pure phase
           rotation: `pos(t+Δt) = exp(+iωΔt) · pos(t)`. This is what the
           ETDRK0 integrator computes exactly.

        After the exponential step, the inverse rotation and unscaling recover
        the updated `(h, v)`.

        At `k = 0` (the spatial mean), the frequency is zero and the two modes
        collapse — the system is no longer diagonalizable. There, the exact
        update is simply `h_mean += Δt · v_mean`, which is applied as a
        separate correction.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `speed_of_sound` (keyword-only): The wave speed `c`. Default: `1.0`.

        **Notes:**

        - The stepper is unconditionally stable, no matter the choice of
            any argument because the equation is solved analytically in Fourier
            space.
        - Ultimately, only the factor `c Δt / L` affects the characteristic
            of the dynamics.
        - The implementation relies on a handcrafted diagonalization of the
          system in Fourier space, which is specific to the wave equation.
          Hence, wave dynamics is not part of the generic steppers like
          [`exponax.stepper.generic.GeneralLinearStepper`][]
        """
        self.speed_of_sound = speed_of_sound
        self.wavenumber_norm = jnp.linalg.norm(
            build_scaled_wavenumbers(
                num_spatial_dims=num_spatial_dims,
                domain_extent=domain_extent,
                num_points=num_points,
            ),
            axis=0,
            keepdims=True,
        )
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=2,
            order=0,
        )

    def _forward_transform(
        self, u_hat: Complex[Array, " 2 ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        """Transform (h, v) into diagonalized forward/backward wave modes."""
        h_hat, v_hat = u_hat[0:1], u_hat[1:2]
        # Scale height to match velocity units: w = i c |k| h
        k_guard = jnp.where(self.wavenumber_norm == 0, 1.0, self.wavenumber_norm)
        w_hat = 1j * self.speed_of_sound * k_guard * h_hat

        # Orthonormal rotation into wave modes
        pos = (1 / jnp.sqrt(2)) * (w_hat + v_hat)
        neg = (1 / jnp.sqrt(2)) * (w_hat - v_hat)
        return jnp.concatenate([pos, neg], axis=0)

    def _inverse_transform(
        self, waves_hat: Complex[Array, " 2 ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        """Transform diagonalized wave modes back into (h, v)."""
        pos, neg = waves_hat[0:1], waves_hat[1:2]
        # Inverse rotation (the rotation matrix is its own inverse)
        w_hat = (1 / jnp.sqrt(2)) * (pos + neg)
        v_hat = (1 / jnp.sqrt(2)) * (pos - neg)

        # Undo scaling to recover height
        k_guard = jnp.where(self.wavenumber_norm == 0, 1.0, self.wavenumber_norm)
        h_hat = w_hat / (1j * self.speed_of_sound * k_guard)
        return jnp.concatenate([h_hat, v_hat], axis=0)

    def _build_linear_operator(
        self, derivative_operator: Complex[Array, " D ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        val = 1j * self.speed_of_sound * self.wavenumber_norm
        return jnp.concatenate(
            (
                val,
                -val,
            ),
            axis=0,
        )

    def _build_nonlinear_fun(
        self, derivative_operator: Complex[Array, " D ... (N//2)+1"]
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(self.num_spatial_dims, self.num_points)

    def step_fourier(
        self, u_hat: Complex[Array, " 2 ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        """
        Advance the state by one timestep in Fourier space.

        Overrides the base method to wrap the ETDRK step with the
        forward/inverse diagonalization transforms.
        """
        waves_hat = self._forward_transform(u_hat)
        waves_hat_next = super().step_fourier(waves_hat)
        u_hat_next = self._inverse_transform(waves_hat_next)

        # The k=0 (mean/DC) mode cannot be diagonalized because the two
        # eigenvalues collapse to zero and the system matrix becomes
        # [[0, 1], [0, 0]]. The diagonalization leaves this mode unchanged,
        # but the exact solution is h_mean(t+dt) = h_mean(t) + dt * v_mean(t).
        # Apply this linear drift explicitly.
        h_dc_idx = (0,) + (0,) * self.num_spatial_dims
        v_dc_idx = (1,) + (0,) * self.num_spatial_dims
        u_hat_next = u_hat_next.at[h_dc_idx].add(self.dt * u_hat[v_dc_idx])

        return u_hat_next
