import jax
import jax.numpy as jnp
import pytest

import exponax as ex

# ---------------------------------------------------------------------------
# Amplitude spectrum (power=False)
# ---------------------------------------------------------------------------


def test_amplitude_spectrum_1d():
    grid = ex.make_grid(1, 2 * jnp.pi, 128)

    u = 3.0 * jnp.sin(grid)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.cos(2 * grid)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 2] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(3 * grid) + 4.0 * jnp.cos(3 * grid)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 3] == pytest.approx(jnp.sqrt(3.0**2 + 4.0**2))

    u = 3.0 * jnp.ones_like(grid)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 0] == pytest.approx(3.0)


def test_amplitude_spectrum_2d():
    grid = ex.make_grid(2, 2 * jnp.pi, 48)

    # Axis-aligned modes
    u = 3.0 * jnp.sin(grid[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.cos(2 * grid[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 2] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(3 * grid[0:1]) + 4.0 * jnp.cos(3 * grid[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 3] == pytest.approx(jnp.sqrt(3.0**2 + 4.0**2))

    u = 3.0 * jnp.ones_like(grid[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 0] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(grid[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.cos(2 * grid[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 2] == pytest.approx(3.0)

    # Mixed 2D modes
    u = 3.0 * jnp.sin(1 * grid[0:1]) * jnp.cos(1 * grid[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(2 * grid[0:1]) * jnp.cos(2 * grid[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    # The amplitude is in the 3-bin because the wavenumber norm of [2, 2] is
    # 2*sqrt(2) = 2.8284 which is not in the interval [1.5, 2.5).
    assert spectrum[0, 3] == pytest.approx(3.0)
    assert spectrum[0, 2] == pytest.approx(0.0, abs=1e-5)


def test_amplitude_spectrum_3d():
    grid = ex.make_grid(3, 2 * jnp.pi, 16)

    # Axis-aligned along each of the three axes
    for axis in range(3):
        u = 3.0 * jnp.sin(grid[axis : axis + 1])
        spectrum = ex.get_spectrum(u, power=False)
        assert spectrum[0, 1] == pytest.approx(3.0, abs=1e-4)

    # Mixed 3D mode: sin(x)*cos(y)*sin(z), wavenumber norm = sqrt(3) ≈ 1.73
    # Falls into bin 2 (interval [1.5, 2.5))
    u = 3.0 * (jnp.sin(grid[0:1]) * jnp.cos(grid[1:2]) * jnp.sin(grid[2:3]))
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 2] == pytest.approx(3.0, abs=1e-4)

    # Constant field
    u = 5.0 * jnp.ones_like(grid[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 0] == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Power spectrum (power=True)
# ---------------------------------------------------------------------------


def test_power_spectrum_1d():
    """Power spectrum for single-mode signals in 1D.

    For a signal u = A*cos(kx), the power spectrum at bin k should be the
    energy contribution of that mode.  The key question is whether the blanket
    0.5 factor is consistent across DC, intermediate, and Nyquist modes.
    """
    N = 16
    grid = ex.make_grid(1, 2 * jnp.pi, N)

    # --- intermediate mode: u = 5*cos(3x) ---
    A = 5.0
    k = 3
    u = A * jnp.cos(k * grid)
    spectrum = ex.get_spectrum(u, power=True)
    energy = 0.5 * jnp.mean(u**2)
    # The only populated bin should account for all the energy
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-5)

    # --- DC mode: u = 3 (constant) ---
    A = 3.0
    u = A * jnp.ones_like(grid)
    spectrum = ex.get_spectrum(u, power=True)
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-5)

    # --- Nyquist mode (even N): u = 2*(-1)^n ---
    A = 2.0
    n = jnp.arange(N)
    u = (A * (-1.0) ** n)[None, :]  # shape (1, N)
    spectrum = ex.get_spectrum(u, power=True)
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-5)


def test_power_spectrum_2d():
    """Power spectrum in 2D for axis-aligned single-mode signals."""
    N = 32
    grid = ex.make_grid(2, 2 * jnp.pi, N)

    # Cosine along first axis
    u = 4.0 * jnp.cos(3 * grid[0:1])
    spectrum = ex.get_spectrum(u, power=True)
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-4)

    # Cosine along second axis (rfft axis)
    u = 4.0 * jnp.cos(3 * grid[1:2])
    spectrum = ex.get_spectrum(u, power=True)
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-4)

    # Constant field
    u = 3.0 * jnp.ones_like(grid[0:1])
    spectrum = ex.get_spectrum(u, power=True)
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-4)


# ---------------------------------------------------------------------------
# Parseval's theorem: sum(spectrum) == 0.5 * mean(u²)
# ---------------------------------------------------------------------------


def test_parseval_1d():
    """Parseval's theorem with shell sum in 1D."""
    N = 64
    grid = ex.make_grid(1, 2 * jnp.pi, N)

    # Multi-mode signal
    u = 2.0 * jnp.cos(grid) + 3.0 * jnp.sin(5 * grid) + 1.0
    spectrum = ex.get_spectrum(u, power=True, radial_binning="sum")
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-5)


def test_parseval_2d():
    """Parseval's theorem with shell sum in 2D."""
    N = 32
    grid = ex.make_grid(2, 2 * jnp.pi, N)

    # Axis-aligned multi-mode
    u = 2.0 * jnp.cos(grid[0:1]) + 3.0 * jnp.sin(5 * grid[1:2]) + 1.0
    spectrum = ex.get_spectrum(u, power=True, radial_binning="sum")
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-4)

    # Mixed 2D mode
    u = 4.0 * jnp.sin(2 * grid[0:1]) * jnp.cos(3 * grid[1:2])
    spectrum = ex.get_spectrum(u, power=True, radial_binning="sum")
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-4)


def test_parseval_2d_random():
    """Parseval's theorem for a random field in 2D.

    A raw random field has energy in the corners of the wavenumber cube
    (|k| > N/2) which falls outside the radial bins.  We low-pass filter
    to the Nyquist sphere first so that all energy is captured by the
    shell sum.
    """
    N = 32
    key = jax.random.PRNGKey(0)
    u_hat = jax.random.normal(key, shape=(1, N, N // 2 + 1)) + 1j * jax.random.normal(
        jax.random.PRNGKey(1), shape=(1, N, N // 2 + 1)
    )
    # Zero out modes outside the Nyquist sphere (|k| > N/2)
    mask = ex._spectral.low_pass_filter_mask(2, N, cutoff=N // 2, axis_separate=False)
    u_hat = u_hat * mask
    u = ex.ifft(u_hat, num_spatial_dims=2, num_points=N)

    spectrum = ex.get_spectrum(u, power=True, radial_binning="sum")
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-4)


def test_parseval_3d():
    """Parseval's theorem with shell sum in 3D."""
    N = 16
    grid = ex.make_grid(3, 2 * jnp.pi, N)

    u = 2.0 * jnp.cos(grid[0:1]) + 3.0 * jnp.sin(2 * grid[2:3]) + 1.0
    spectrum = ex.get_spectrum(u, power=True, radial_binning="sum")
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-3)


def test_parseval_3d_random():
    """Parseval's theorem for a random field in 3D.

    A raw random field has energy in the corners of the wavenumber cube
    (|k| > N/2) which falls outside the radial bins.  We low-pass filter
    to the Nyquist sphere first so that all energy is captured by the
    shell sum.
    """
    N = 16
    key = jax.random.PRNGKey(42)
    u_hat = jax.random.normal(
        key, shape=(1, N, N, N // 2 + 1)
    ) + 1j * jax.random.normal(jax.random.PRNGKey(43), shape=(1, N, N, N // 2 + 1))
    # Zero out modes outside the Nyquist sphere (|k| > N/2)
    mask = ex._spectral.low_pass_filter_mask(3, N, cutoff=N // 2, axis_separate=False)
    u_hat = u_hat * mask
    u = ex.ifft(u_hat, num_spatial_dims=3, num_points=N)

    spectrum = ex.get_spectrum(u, power=True, radial_binning="sum")
    energy = 0.5 * jnp.mean(u**2)
    assert float(jnp.sum(spectrum)) == pytest.approx(float(energy), rel=1e-3)
