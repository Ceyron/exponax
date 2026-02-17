import jax
import jax.numpy as jnp
import pytest

import exponax as ex
from exponax._spectral import fft, ifft
from exponax.stepper import Wave

L = 2 * jnp.pi
PI = jnp.pi


# ===========================================================================
# Instantiation
# ===========================================================================


class TestWaveInstantiation:
    @pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
    def test_instantiate(self, num_spatial_dims):
        stepper = Wave(num_spatial_dims, 10.0, 25, 0.1)
        assert stepper.num_channels == 2
        assert stepper.num_spatial_dims == num_spatial_dims

    @pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
    def test_output_shape(self, num_spatial_dims):
        N = 16
        stepper = Wave(num_spatial_dims, L, N, 0.01)
        u0 = jnp.zeros((2,) + (N,) * num_spatial_dims)
        u1 = stepper(u0)
        assert u1.shape == u0.shape
        assert jnp.all(jnp.isfinite(u1))

    def test_wrong_input_shape_raises(self):
        stepper = Wave(1, L, 32, 0.01)
        with pytest.raises(ValueError, match="Expected shape"):
            stepper(jnp.zeros((1, 32)))  # needs 2 channels

    def test_default_speed_of_sound(self):
        stepper = Wave(1, L, 32, 0.01)
        assert stepper.speed_of_sound == 1.0

    def test_custom_speed_of_sound(self):
        stepper = Wave(1, L, 32, 0.01, speed_of_sound=3.5)
        assert stepper.speed_of_sound == 3.5


# ===========================================================================
# Analytical correctness — 1D standing wave
# ===========================================================================


class TestWaveAnalytical1D:
    """Exact solutions for h(x,0) = cos(k0 x), v(x,0) = 0, c = speed:
    h(x,t) = cos(k0 x) cos(c k0 t)
    v(x,t) = -c k0 cos(k0 x) sin(c k0 t)

    Note: this uses the natural wavenumber k0 on domain [0, 2pi].
    """

    def _make_stepper_and_ic(self, k0, c=1.0, N=64, dt=0.01):
        stepper = Wave(1, L, N, dt, speed_of_sound=c)
        x = jnp.linspace(0, L, N, endpoint=False)
        h0 = jnp.cos(k0 * x)[None]
        v0 = jnp.zeros_like(h0)
        u0 = jnp.concatenate([h0, v0], axis=0)
        return stepper, x, u0

    @pytest.mark.parametrize("k0", [1, 2, 3, 5, 10])
    def test_single_mode(self, k0):
        """A single Fourier mode should oscillate at the correct frequency."""
        c, N, dt = 1.0, 64, 0.01
        stepper, x, u0 = self._make_stepper_and_ic(k0, c, N, dt)

        # Few steps to keep float32 accumulation small
        n_steps = 10
        u = u0
        for _ in range(n_steps):
            u = stepper.step(u)
        t = n_steps * dt

        h_exact = jnp.cos(k0 * x) * jnp.cos(c * k0 * t)
        v_exact = -c * k0 * jnp.cos(k0 * x) * jnp.sin(c * k0 * t)

        assert u[0] == pytest.approx(h_exact, abs=1e-4)
        assert u[1] == pytest.approx(v_exact, abs=1e-3)

    def test_full_period_return(self):
        """After one full period T = 2*pi/(c*k0), state returns to initial."""
        k0, c, N = 3, 1.0, 64
        n_steps = 200
        T = 2 * PI / (c * k0)
        dt = float(T / n_steps)  # dt exactly divides the period
        stepper, _, u0 = self._make_stepper_and_ic(k0, c, N, dt)

        u = u0
        for _ in range(n_steps):
            u = stepper.step(u)

        assert u[0] == pytest.approx(u0[0], abs=1e-3)
        assert u[1] == pytest.approx(u0[1], abs=1e-3)

    def test_superposition(self):
        """Linearity: h = cos(x) + 0.5*cos(3x) should evolve as sum of modes."""
        c, N, dt = 1.0, 64, 0.01
        stepper = Wave(1, L, N, dt, speed_of_sound=c)
        x = jnp.linspace(0, L, N, endpoint=False)

        h0 = (jnp.cos(x) + 0.5 * jnp.cos(3 * x))[None]
        v0 = jnp.zeros_like(h0)
        u0 = jnp.concatenate([h0, v0], axis=0)

        n_steps = 10
        u = u0
        for _ in range(n_steps):
            u = stepper.step(u)
        t = n_steps * dt

        h_exact = jnp.cos(x) * jnp.cos(c * t) + 0.5 * jnp.cos(3 * x) * jnp.cos(
            c * 3 * t
        )
        assert u[0] == pytest.approx(h_exact, abs=1e-4)

    @pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
    def test_speed_of_sound_scaling(self, c):
        """Oscillation frequency should scale linearly with c."""
        k0, N, dt = 2, 64, 0.01
        stepper, x, u0 = self._make_stepper_and_ic(k0, c, N, dt)

        n_steps = 10
        u = u0
        for _ in range(n_steps):
            u = stepper.step(u)
        t = n_steps * dt

        h_exact = jnp.cos(k0 * x) * jnp.cos(c * k0 * t)
        assert u[0] == pytest.approx(h_exact, abs=1e-4)


# ===========================================================================
# DC mode (k=0) — mean height drift
# ===========================================================================


class TestWaveDCMode:
    def test_dc_drift_single_step_1d(self):
        """Nonzero mean velocity should cause mean height to drift by dt*v_mean."""
        dt = 0.05
        N = 64
        stepper = Wave(1, L, N, dt, speed_of_sound=1.0)
        x = jnp.linspace(0, L, N, endpoint=False)
        h0 = jnp.cos(x)[None]
        v0 = jnp.ones((1, N))
        u0 = jnp.concatenate([h0, v0], axis=0)

        u1 = stepper.step(u0)
        expected_h_mean = float(jnp.mean(h0)) + dt * 1.0
        assert float(jnp.mean(u1[0])) == pytest.approx(expected_h_mean, abs=1e-5)

    def test_dc_drift_accumulates(self):
        """Over 100 steps, mean height should grow linearly."""
        dt = 0.05
        N = 64
        stepper = Wave(1, L, N, dt, speed_of_sound=1.0)
        v_mean = 2.0
        u = jnp.concatenate([jnp.zeros((1, N)), jnp.full((1, N), v_mean)], axis=0)

        n_steps = 100
        for _ in range(n_steps):
            u = stepper.step(u)

        expected_h_mean = n_steps * dt * v_mean
        assert float(jnp.mean(u[0])) == pytest.approx(expected_h_mean, abs=1e-3)

    def test_dc_drift_2d(self):
        """DC drift should work correctly in 2D."""
        dt = 0.01
        N = 16
        stepper = Wave(2, L, N, dt, speed_of_sound=1.0)
        v_mean = 3.0
        u0 = jnp.concatenate(
            [jnp.zeros((1, N, N)), jnp.full((1, N, N), v_mean)], axis=0
        )

        u1 = stepper.step(u0)
        expected_h_mean = dt * v_mean
        assert float(jnp.mean(u1[0])) == pytest.approx(expected_h_mean, abs=1e-5)

    def test_zero_mean_velocity_no_drift(self):
        """With zero mean velocity, mean height should stay constant."""
        dt = 0.05
        N = 64
        stepper = Wave(1, L, N, dt, speed_of_sound=1.0)
        x = jnp.linspace(0, L, N, endpoint=False)
        h0 = jnp.cos(x)[None]
        v0 = jnp.sin(x)[None]  # zero mean
        u0 = jnp.concatenate([h0, v0], axis=0)

        u1 = stepper.step(u0)
        assert float(jnp.mean(u1[0])) == pytest.approx(float(jnp.mean(h0)), abs=1e-5)


# ===========================================================================
# Energy conservation
# ===========================================================================


class TestWaveEnergyConservation:
    @staticmethod
    def _wave_energy(u, c, domain_extent):
        """E = 0.5 * integral(v^2 + c^2 |grad h|^2) dx."""
        h, v = u[0:1], u[1:2]
        grad_h = ex.derivative(h, domain_extent)
        dx = domain_extent / u.shape[-1]
        return 0.5 * float(
            jnp.sum(v**2 + c**2 * jnp.sum(grad_h**2, axis=0, keepdims=True)) * dx
        )

    def test_energy_conservation_1d(self):
        """Wave equation should conserve energy over many steps."""
        c = 1.0
        stepper = Wave(1, L, 64, 0.01, speed_of_sound=c)
        x = jnp.linspace(0, L, 64, endpoint=False)
        u = jnp.stack([jnp.cos(3 * x), jnp.zeros(64)])

        e0 = self._wave_energy(u, c, L)
        for _ in range(1000):
            u = stepper.step(u)
        e_final = self._wave_energy(u, c, L)

        assert e_final == pytest.approx(e0, rel=1e-3)

    def test_energy_conservation_2d(self):
        """Energy conservation in 2D."""
        c, N, dt = 1.0, 32, 0.01
        stepper = Wave(2, L, N, dt, speed_of_sound=c)
        x = jnp.linspace(0, L, N, endpoint=False)
        xx, yy = jnp.meshgrid(x, x, indexing="ij")
        h0 = jnp.cos(2 * xx + 3 * yy)[None]
        u = jnp.concatenate([h0, jnp.zeros_like(h0)], axis=0)

        e0 = self._wave_energy(u, c, L)
        for _ in range(200):
            u = stepper.step(u)
        e_final = self._wave_energy(u, c, L)

        assert e_final == pytest.approx(e0, rel=1e-3)


# ===========================================================================
# Long-time stability
# ===========================================================================


class TestWaveStability:
    def test_long_time_bounded(self):
        """Solution should remain bounded over 10000 steps."""
        stepper = Wave(1, L, 64, 0.01, speed_of_sound=1.0)
        x = jnp.linspace(0, L, 64, endpoint=False)
        u = jnp.stack([jnp.cos(3 * x), jnp.zeros(64)])

        for _ in range(10000):
            u = stepper.step(u)

        assert jnp.all(jnp.isfinite(u))
        assert float(jnp.max(jnp.abs(u))) < 100.0


# ===========================================================================
# 2D analytical
# ===========================================================================


class TestWaveAnalytical2D:
    def test_2d_plane_wave(self):
        """2D plane wave h(x,y,0) = cos(k1*x + k2*y), v=0 should oscillate
        at omega = c * sqrt(k1^2 + k2^2)."""
        c, N, dt = 1.0, 32, 0.01
        k1, k2 = 2, 3
        stepper = Wave(2, L, N, dt, speed_of_sound=c)

        x = jnp.linspace(0, L, N, endpoint=False)
        xx, yy = jnp.meshgrid(x, x, indexing="ij")
        h0 = jnp.cos(k1 * xx + k2 * yy)[None]
        u0 = jnp.concatenate([h0, jnp.zeros_like(h0)], axis=0)

        n_steps = 10
        u = u0
        for _ in range(n_steps):
            u = stepper.step(u)
        t = n_steps * dt

        omega = c * jnp.sqrt(float(k1**2 + k2**2))
        h_exact = jnp.cos(k1 * xx + k2 * yy) * jnp.cos(omega * t)

        assert u[0] == pytest.approx(h_exact, abs=1e-4)


# ===========================================================================
# Ecosystem compatibility
# ===========================================================================


class TestWaveEcosystem:
    def test_rollout(self):
        """exponax.rollout should work with Wave stepper."""
        stepper = Wave(1, L, 64, 0.01)
        x = jnp.linspace(0, L, 64, endpoint=False)
        u0 = jnp.stack([jnp.cos(x), jnp.zeros(64)])

        trajectory = ex.rollout(stepper, 10, include_init=True)(u0)
        assert trajectory.shape == (11, 2, 64)
        assert jnp.all(jnp.isfinite(trajectory))

    def test_vmap(self):
        """jax.vmap should work for batched execution."""
        stepper = Wave(1, L, 64, 0.01)
        x = jnp.linspace(0, L, 64, endpoint=False)
        u0 = jnp.stack([jnp.cos(x), jnp.zeros(64)])
        batch = jnp.stack([u0, u0 * 2, u0 * 0.5])

        results = jax.vmap(stepper)(batch)
        assert results.shape == (3, 2, 64)
        assert jnp.all(jnp.isfinite(results))

    def test_step_fourier_consistency(self):
        """step(u) should equal fft -> step_fourier -> ifft."""
        N = 64
        stepper = Wave(1, L, N, 0.01)
        x = jnp.linspace(0, L, N, endpoint=False)
        u0 = jnp.stack([jnp.cos(3 * x), jnp.sin(x)])

        u_hat = fft(u0, num_spatial_dims=1)
        u_next_hat = stepper.step_fourier(u_hat)
        u_next_manual = ifft(u_next_hat, num_spatial_dims=1, num_points=N)
        u_next_step = stepper.step(u0)

        assert u_next_manual == pytest.approx(u_next_step, abs=1e-5)

    def test_repeated_stepper(self):
        """Wave inside RepeatedStepper should match manual sub-stepping."""
        N, dt, n_sub = 64, 0.01, 5
        sub_stepper = Wave(1, L, N, dt)
        repeated = ex.RepeatedStepper(sub_stepper, n_sub)

        x = jnp.linspace(0, L, N, endpoint=False)
        u0 = jnp.stack([jnp.cos(3 * x), jnp.zeros(N)])

        u_repeated = repeated(u0)

        u_manual = u0
        for _ in range(n_sub):
            u_manual = sub_stepper.step(u_manual)

        assert u_repeated == pytest.approx(u_manual, abs=1e-4)
