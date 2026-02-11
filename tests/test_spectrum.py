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


# ---------------------------------------------------------------------------
# Radial binning: "average" vs "sum"
# ---------------------------------------------------------------------------


def test_binning_average_vs_sum_1d():
    """In 1D there is no radial binning, so average and sum should agree."""
    N = 64
    grid = ex.make_grid(1, 2 * jnp.pi, N)
    u = 3.0 * jnp.sin(grid) + 2.0 * jnp.cos(5 * grid)

    spec_sum = ex.get_spectrum(u, power=True, radial_binning="sum")
    spec_avg = ex.get_spectrum(u, power=True, radial_binning="average")
    assert jnp.allclose(spec_sum, spec_avg)


def test_binning_average_vs_sum_2d():
    """Verify that sum = mode_count * average for each radial bin in 2D.

    The mode count per bin is computed directly from the wavenumber mesh and
    should approximately scale as 2*pi*k (shell circumference) for large k.
    """
    N = 32

    # Compute the exact mode count per radial bin from the wavenumber mesh
    wavenumbers_mesh = ex._spectral.build_wavenumbers(2, N)
    wavenumbers_norm = jnp.linalg.norm(wavenumbers_mesh, axis=0)
    wavenumbers_1d = jnp.arange(N // 2 + 1, dtype=float)

    mode_count = jnp.zeros(N // 2 + 1)
    for i, k in enumerate(wavenumbers_1d):
        mask = (wavenumbers_norm >= k - 0.5) & (wavenumbers_norm < k + 0.5)
        mode_count = mode_count.at[i].set(jnp.sum(mask))

    # Use a low-pass filtered random field so all bins within the Nyquist
    # sphere are populated
    key = jax.random.PRNGKey(0)
    u_hat = jax.random.normal(key, shape=(1, N, N // 2 + 1)) + 1j * jax.random.normal(
        jax.random.PRNGKey(1), shape=(1, N, N // 2 + 1)
    )
    mask = ex._spectral.low_pass_filter_mask(2, N, cutoff=N // 2, axis_separate=False)
    u_hat = u_hat * mask
    u = ex.ifft(u_hat, num_spatial_dims=2, num_points=N)

    spec_sum = ex.get_spectrum(u, power=True, radial_binning="sum")
    spec_avg = ex.get_spectrum(u, power=True, radial_binning="average")

    # For every bin with modes, verify sum == mode_count * average
    for k in range(N // 2 + 1):
        if mode_count[k] > 0 and float(spec_avg[0, k]) > 1e-10:
            ratio = float(spec_sum[0, k] / spec_avg[0, k])
            assert ratio == pytest.approx(float(mode_count[k]), rel=1e-4), (
                f"Bin {k}: ratio={ratio}, expected mode_count={int(mode_count[k])}"
            )

    # Sanity check: at large k the mode count should approach pi*k
    # (half of the full 2*pi*k because the rfft grid only stores k_last >= 0)
    large_k = jnp.arange(5, N // 4)
    expected_surface = jnp.pi * large_k
    actual_counts = mode_count[5 : N // 4]
    assert jnp.allclose(actual_counts, expected_surface, rtol=0.2)


def test_binning_average_vs_sum_3d():
    """Verify that sum = mode_count * average for each radial bin in 3D.

    The mode count per bin should approximately scale as 2*pi*k² (half of the
    full 4*pi*k² shell surface area because the rfft grid only stores
    k_last >= 0).
    """
    N = 16

    # Compute the exact mode count per radial bin from the wavenumber mesh
    wavenumbers_mesh = ex._spectral.build_wavenumbers(3, N)
    wavenumbers_norm = jnp.linalg.norm(wavenumbers_mesh, axis=0)
    wavenumbers_1d = jnp.arange(N // 2 + 1, dtype=float)

    mode_count = jnp.zeros(N // 2 + 1)
    for i, k in enumerate(wavenumbers_1d):
        bin_mask = (wavenumbers_norm >= k - 0.5) & (wavenumbers_norm < k + 0.5)
        mode_count = mode_count.at[i].set(jnp.sum(bin_mask))

    # Use a low-pass filtered random field so all bins are populated
    key = jax.random.PRNGKey(10)
    u_hat = jax.random.normal(
        key, shape=(1, N, N, N // 2 + 1)
    ) + 1j * jax.random.normal(jax.random.PRNGKey(11), shape=(1, N, N, N // 2 + 1))
    mask = ex._spectral.low_pass_filter_mask(3, N, cutoff=N // 2, axis_separate=False)
    u_hat = u_hat * mask
    u = ex.ifft(u_hat, num_spatial_dims=3, num_points=N)

    spec_sum = ex.get_spectrum(u, power=True, radial_binning="sum")
    spec_avg = ex.get_spectrum(u, power=True, radial_binning="average")

    # For every bin with modes, verify sum == mode_count * average
    for k in range(N // 2 + 1):
        if mode_count[k] > 0 and float(spec_avg[0, k]) > 1e-10:
            ratio = float(spec_sum[0, k] / spec_avg[0, k])
            assert ratio == pytest.approx(float(mode_count[k]), rel=1e-3), (
                f"Bin {k}: ratio={ratio}, expected mode_count={int(mode_count[k])}"
            )

    # Sanity check: at large k the mode count should approach 2*pi*k²
    # (half of the full 4*pi*k² because the rfft grid only stores k_last >= 0)
    large_k = jnp.arange(3, N // 4)
    expected_surface = 2 * jnp.pi * large_k**2
    actual_counts = mode_count[3 : N // 4]
    assert jnp.allclose(actual_counts, expected_surface, rtol=0.25)


# ---------------------------------------------------------------------------
# rfft compensation: spectrum should be the same regardless of which spatial
# axis carries the mode (the rfft axis vs. the full-fft axes)
# ---------------------------------------------------------------------------


def test_rfft_axis_symmetry_2d():
    """A cosine along axis 0 (full fft) and axis 1 (rfft) with the same
    wavenumber should produce the same spectrum."""
    N = 32
    grid = ex.make_grid(2, 2 * jnp.pi, N)

    u_axis0 = 4.0 * jnp.cos(3 * grid[0:1])
    u_axis1 = 4.0 * jnp.cos(3 * grid[1:2])

    spec0 = ex.get_spectrum(u_axis0, power=True)
    spec1 = ex.get_spectrum(u_axis1, power=True)
    assert jnp.allclose(spec0, spec1, atol=1e-5)


def test_rfft_axis_symmetry_3d():
    """Same test in 3D: a cosine along each of the three axes should give
    identical spectra."""
    N = 16
    grid = ex.make_grid(3, 2 * jnp.pi, N)

    spectra = []
    for axis in range(3):
        u = 4.0 * jnp.cos(3 * grid[axis : axis + 1])
        spectra.append(ex.get_spectrum(u, power=True))

    assert jnp.allclose(spectra[0], spectra[1], atol=1e-4)
    assert jnp.allclose(spectra[0], spectra[2], atol=1e-4)


# ---------------------------------------------------------------------------
# Multi-channel
# ---------------------------------------------------------------------------


def test_multi_channel():
    """Each channel should get its own independent spectrum."""
    N = 64
    grid = ex.make_grid(1, 2 * jnp.pi, N)

    # Channel 0: amplitude 3 at wavenumber 2
    # Channel 1: amplitude 5 at wavenumber 7
    u = jnp.concatenate([3.0 * jnp.cos(2 * grid), 5.0 * jnp.sin(7 * grid)], axis=0)
    assert u.shape == (2, N)

    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum.shape == (2, N // 2 + 1)
    assert spectrum[0, 2] == pytest.approx(3.0)
    assert spectrum[1, 7] == pytest.approx(5.0)
    # Cross-check: channel 0 should have nothing at wavenumber 7 and vice versa
    assert spectrum[0, 7] == pytest.approx(0.0, abs=1e-5)
    assert spectrum[1, 2] == pytest.approx(0.0, abs=1e-5)


def test_multi_channel_2d():
    """Multi-channel in 2D."""
    N = 32
    grid = ex.make_grid(2, 2 * jnp.pi, N)

    u = jnp.concatenate(
        [3.0 * jnp.cos(2 * grid[0:1]), 5.0 * jnp.sin(4 * grid[1:2])], axis=0
    )
    assert u.shape == (2, N, N)

    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum.shape == (2, N // 2 + 1)
    assert spectrum[0, 2] == pytest.approx(3.0, abs=1e-4)
    assert spectrum[1, 4] == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_field():
    """Spectrum of a zero field should be all zeros."""
    for ndim in [1, 2, 3]:
        N = 16
        shape = (1,) + (N,) * ndim
        u = jnp.zeros(shape)
        spectrum = ex.get_spectrum(u, power=True)
        assert jnp.allclose(spectrum, 0.0)
        spectrum = ex.get_spectrum(u, power=False)
        assert jnp.allclose(spectrum, 0.0)


def test_output_shape():
    """Verify spectrum output shape is (C, N//2+1) regardless of spatial dims."""
    for ndim in [1, 2, 3]:
        N = 16
        C = 3
        shape = (C,) + (N,) * ndim
        u = jnp.ones(shape)
        spectrum = ex.get_spectrum(u, power=True)
        assert spectrum.shape == (C, N // 2 + 1)
