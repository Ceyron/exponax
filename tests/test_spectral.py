import jax.numpy as jnp
import pytest

import exponax as ex
from exponax._spectral import (
    build_derivative_operator,
    build_laplace_operator,
    build_scaled_wavenumbers,
    fft,
    ifft,
    make_incompressible,
)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_fft_ifft_roundtrip(num_spatial_dims):
    num_points = 32
    domain_extent = 5.0

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u = jnp.sin(2 * jnp.pi * grid[0:1] / domain_extent)

    u_hat = fft(u, num_spatial_dims=num_spatial_dims)
    u_reconstructed = ifft(
        u_hat, num_spatial_dims=num_spatial_dims, num_points=num_points
    )

    assert u_reconstructed == pytest.approx(u, abs=1e-5)


@pytest.mark.parametrize(
    "num_spatial_dims,derivative_axis",
    [
        (1, 0),
        (2, 0),
        (2, 1),
    ],
)
def test_derivative_of_sine(num_spatial_dims, derivative_axis):
    """Spectral derivative of sin(2*pi*k*x/L) should give (2*pi*k/L)*cos(...)."""
    num_points = 64
    domain_extent = 3.0
    k = 3  # wavenumber

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u = jnp.sin(
        k * 2 * jnp.pi * grid[derivative_axis : derivative_axis + 1] / domain_extent
    )

    u_der = ex.derivative(u, domain_extent, order=1)

    expected = (
        k
        * 2
        * jnp.pi
        / domain_extent
        * jnp.cos(
            k * 2 * jnp.pi * grid[derivative_axis : derivative_axis + 1] / domain_extent
        )
    )

    # For 1D (single channel input), derivative returns shape (D, ..., N) For
    # multi-D, it also returns (D, ..., N). We compare the derivative_axis
    # component.
    assert u_der[derivative_axis] == pytest.approx(expected[0], abs=1e-4)


def test_derivative_second_order():
    """Second derivative of sin should give -k^2 * sin."""
    num_points = 64
    domain_extent = 2 * jnp.pi
    k = 3

    grid = ex.make_grid(1, domain_extent, num_points)
    u = jnp.sin(k * grid)

    u_der2 = ex.derivative(u, domain_extent, order=2)

    expected = -(k**2) * jnp.sin(k * grid)

    assert u_der2 == pytest.approx(expected, abs=1e-3)


def test_laplace_operator_eigenvalues():
    """Laplacian eigenvalue for wavenumber k should be -(2*pi*k/L)^2."""
    num_points = 32
    domain_extent = 4.0

    derivative_operator = build_derivative_operator(1, domain_extent, num_points)
    laplace_op = build_laplace_operator(derivative_operator)

    scaled_wavenumbers = build_scaled_wavenumbers(1, domain_extent, num_points)

    expected_eigenvalues = -(scaled_wavenumbers[0:1] ** 2)

    assert laplace_op == pytest.approx(expected_eigenvalues, abs=1e-10)


def test_make_incompressible_preserves_solenoidal():
    """An already divergence-free field should be preserved by the projection."""
    num_points = 32
    domain_extent = 2 * jnp.pi

    grid = ex.make_grid(2, domain_extent, num_points)
    # Stream function psi = sin(x)*sin(y) gives div-free field:
    # vx = -dpsi/dy = -sin(x)*cos(y), vy = dpsi/dx = cos(x)*sin(y)
    vx = -jnp.sin(grid[0:1]) * jnp.cos(grid[1:2])
    vy = jnp.cos(grid[0:1]) * jnp.sin(grid[1:2])
    velocity = jnp.concatenate([vx, vy], axis=0)

    velocity_proj = make_incompressible(velocity)

    assert velocity_proj == pytest.approx(velocity, abs=1e-5)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_build_derivative_operator_shape(num_spatial_dims):
    num_points = 16
    domain_extent = 1.0

    deriv_op = build_derivative_operator(num_spatial_dims, domain_extent, num_points)

    expected_shape = (
        (num_spatial_dims,)
        + (num_points,) * (num_spatial_dims - 1)
        + (num_points // 2 + 1,)
    )

    assert deriv_op.shape == expected_shape
