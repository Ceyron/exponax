import jax.numpy as jnp
import pytest

import exponax as ex


@pytest.mark.parametrize(
    "num_spatial_dims",
    [1, 2, 3],
)
def test_wrap_bc(num_spatial_dims):
    domain_extent = 3.0
    num_points = 10

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    full_grid = ex.make_grid(num_spatial_dims, domain_extent, num_points, full=True)

    u = jnp.sin(2 * jnp.pi * grid[0:1] / domain_extent)
    full_u = jnp.sin(2 * jnp.pi * full_grid[0:1] / domain_extent)
    wrapped_u = ex.wrap_bc(u)

    assert wrapped_u == pytest.approx(full_u, abs=1e-5)
