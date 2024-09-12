import jax.numpy as jnp
import pytest

import exponax as ex


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_constant_offset(num_spatial_dims: int):
    DOMAIN_EXTENT = 5.0
    NUM_POINTS = 40
    grid = ex.make_grid(num_spatial_dims, DOMAIN_EXTENT, NUM_POINTS)

    u_0 = 2.0 * jnp.ones_like(grid)
    u_1 = 4.0 * jnp.ones_like(grid)

    assert ex.metrics.MSE(u_1, u_0, domain_extent=1.0) == pytest.approx(4.0)
    assert ex.metrics.MSE(u_1, u_0, domain_extent=DOMAIN_EXTENT) == pytest.approx(
        4.0 / DOMAIN_EXTENT**num_spatial_dims
    )

    # MSE metric is symmetric
    assert ex.metrics.MSE(u_0, u_1, domain_extent=1.0) == ex.metrics.MSE(
        u_1, u_0, domain_extent=1.0
    )
    assert ex.metrics.MSE(u_0, u_1, domain_extent=DOMAIN_EXTENT) == ex.metrics.MSE(
        u_1, u_0, domain_extent=DOMAIN_EXTENT
    )

    # == approx(1.0)
    assert ex.metrics.nMSE(u_1, u_0) == pytest.approx((4.0 - 2.0) ** 2 / (2.0) ** 2)
    assert ex.metrics.nMSE(u_1, u_0) == pytest.approx(1.0)

    # == approx (1/4)
    assert ex.metrics.nMSE(u_0, u_1) == pytest.approx((2.0 - 4.0) ** 2 / (4.0) ** 2)
    assert ex.metrics.nMSE(u_0, u_1) == pytest.approx(1 / 4)

    assert ex.metrics.RMSE(u_1, u_0, domain_extent=1.0) == pytest.approx(2.0)
    assert ex.metrics.RMSE(u_1, u_0, domain_extent=DOMAIN_EXTENT) == pytest.approx(
        jnp.sqrt(4.0 / DOMAIN_EXTENT**num_spatial_dims)
    )

    # RMSE is symmetric
    assert ex.metrics.RMSE(u_0, u_1, domain_extent=1.0) == ex.metrics.RMSE(
        u_1, u_0, domain_extent=1.0
    )
    assert ex.metrics.RMSE(u_0, u_1, domain_extent=DOMAIN_EXTENT) == ex.metrics.RMSE(
        u_1, u_0, domain_extent=DOMAIN_EXTENT
    )

    # == approx(1.0)
    assert ex.metrics.nRMSE(u_1, u_0) == pytest.approx(
        jnp.sqrt((4.0 - 2.0) ** 2 / 2.0**2)
    )
    assert ex.metrics.nRMSE(u_1, u_0) == pytest.approx(1.0)

    # == approx(sqrt(1/4)) == approx(0.5)
    assert ex.metrics.nRMSE(u_0, u_1) == pytest.approx(
        jnp.sqrt((2.0 - 4.0) ** 2 / 4.0**2)
    )
    assert ex.metrics.nRMSE(u_0, u_1) == pytest.approx(0.5)

    # The Fourier nRMSE should be identical to the spatial nRMSE
    # assert ex.metrics.fourier_nRMSE(u_1, u_0) == ex.metrics.nRMSE(u_1, u_0)
    # assert ex.metrics.fourier_nRMSE(u_0, u_1) == ex.metrics.nRMSE(u_0, u_1)
