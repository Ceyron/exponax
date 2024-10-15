import jax
import jax.numpy as jnp
import pytest

import exponax as ex


@pytest.mark.parametrize(
    "num_spatial_dims,num_points", [(D, N) for D in [1, 2, 3] for N in [10, 11]]
)
def test_building_scaling_array_for_norm_compensation(
    num_spatial_dims: int, num_points: int
):
    noise = jax.random.normal(
        jax.random.PRNGKey(0), (1,) + (num_points,) * num_spatial_dims
    )

    noise_hat_norm_backward = jnp.fft.rfftn(
        noise,
        axes=ex.spectral.space_indices(num_spatial_dims),
    )
    noise_hat_norm_forward = jnp.fft.rfftn(
        noise,
        axes=ex.spectral.space_indices(num_spatial_dims),
        norm="forward",
    )

    scaling_array = ex.spectral.build_scaling_array(
        num_spatial_dims,
        num_points,
        mode="norm_compensation",
    )

    noise_hat_norm_backward_scaled = noise_hat_norm_backward / scaling_array

    assert noise_hat_norm_backward_scaled == pytest.approx(noise_hat_norm_forward)


# Mode "reconstruction" is already tested as part of the `test_interpolation.py``


def test_building_scaling_array_for_coef_extraction():
    # 1D
    grid_1d = ex.make_grid(1, 2 * jnp.pi, 10)

    u = 3 * jnp.cos(2 * grid_1d)
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        1,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.ones_like(grid_1d)
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        1,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 0] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.sin(2 * grid_1d)
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        1,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2] == pytest.approx(0.0 - 3.0j)

    u = 3.0 * jnp.cos(5 * grid_1d)
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        1,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 5] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.sin(5 * grid_1d)
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        1,
        10,
        mode="coef_extraction",
    )
    # Nyquist mode sine cannot be captured
    assert u_hat_scaled.round(5)[0, 5] == pytest.approx(0.0 + 0.0j)

    # 2D - single terms
    grid_2d = ex.make_grid(2, 2 * jnp.pi, 10)

    u = 3.0 * jnp.cos(2 * grid_2d[0:1])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2, 0] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.cos(2 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 0, 2] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.ones_like(grid_2d[0:1])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 0, 0] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.sin(2 * grid_2d[0:1])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2, 0] == pytest.approx(0.0 - 3.0j)

    u = 3.0 * jnp.sin(2 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 0, 2] == pytest.approx(0.0 - 3.0j)

    u = 3.0 * jnp.cos(5 * grid_2d[0:1])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 5, 0] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.cos(5 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 0, 5] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.sin(5 * grid_2d[0:1])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    # Nyquist mode sine cannot be captured
    assert u_hat_scaled.round(5)[0, 5, 0] == pytest.approx(0.0 + 0.0j)

    u = 3.0 * jnp.sin(5 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    # Nyquist mode sine cannot be captured
    assert u_hat_scaled.round(5)[0, 0, 5] == pytest.approx(0.0 + 0.0j)

    # 2D - mixed terms
    u = 3.0 * jnp.cos(2 * grid_2d[0:1]) * jnp.cos(2 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2, 2] == pytest.approx(3.0 + 0.0j)

    u = 3.0 * jnp.sin(2 * grid_2d[0:1]) * jnp.sin(2 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2, 2] == pytest.approx(-3.0 + 0.0j)

    u = 3.0 * jnp.cos(2 * grid_2d[0:1]) * jnp.sin(2 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2, 2] == pytest.approx(0.0 - 3.0j)

    u = 3.0 * jnp.sin(2 * grid_2d[0:1]) * jnp.cos(2 * grid_2d[1:2])
    u_hat = ex.fft(u)
    u_hat_scaled = u_hat / ex.spectral.build_scaling_array(
        2,
        10,
        mode="coef_extraction",
    )
    assert u_hat_scaled.round(5)[0, 2, 2] == pytest.approx(0.0 - 3.0j)

    # TODO: 3D
