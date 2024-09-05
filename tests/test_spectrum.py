import jax.numpy as jnp
import pytest

import exponax as ex


def test_amplitude_spectrum():
    # 1D
    grid_1d = ex.make_grid(1, 2 * jnp.pi, 128)

    u = 3.0 * jnp.sin(grid_1d)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.cos(2 * grid_1d)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 2] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(3 * grid_1d) + 4.0 * jnp.cos(3 * grid_1d)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 3] == pytest.approx(jnp.sqrt(3.0**2 + 4.0**2))

    u = 3.0 * jnp.ones_like(grid_1d)
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 0] == pytest.approx(3.0)

    # 2D - single terms
    grid_2d = ex.make_grid(2, 2 * jnp.pi, 48)

    u = 3.0 * jnp.sin(grid_2d[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.cos(2 * grid_2d[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 2] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(3 * grid_2d[0:1]) + 4.0 * jnp.cos(3 * grid_2d[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 3] == pytest.approx(jnp.sqrt(3.0**2 + 4.0**2))

    u = 3.0 * jnp.ones_like(grid_2d[0:1])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 0] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(grid_2d[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.cos(2 * grid_2d[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 2] == pytest.approx(3.0)

    # 2D - mixed terms
    u = 3.0 * jnp.sin(1 * grid_2d[0:1]) * jnp.cos(1 * grid_2d[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    assert spectrum[0, 1] == pytest.approx(3.0)

    u = 3.0 * jnp.sin(2 * grid_2d[0:1]) * jnp.cos(2 * grid_2d[1:2])
    spectrum = ex.get_spectrum(u, power=False)
    # The amplitude is in the 3-bin because the wavenumber norm of [2, 2] is
    # 2*sqrt(2) = 2.8284 which is not in the interval [1.5, 2.5).
    assert spectrum[0, 3] == pytest.approx(3.0)
    assert spectrum[0, 2] == pytest.approx(0.0, abs=1e-5)

    # TODO: 3D
