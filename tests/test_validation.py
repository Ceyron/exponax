import jax.numpy as jnp
import pytest

import exponax as ex

# Linear steppers

# linear steppers do not make spatial and temporal truncation errors, hence we
# can directly compare them with the analytical solution without performing a
# convergence study


def test_advection_1d():
    num_spatial_dims = 1
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    velocity = 0.1

    analytical_solution = lambda t, x: jnp.sin(
        4 * 2 * jnp.pi * (x - velocity * t) / domain_extent
    )

    grid = ex.get_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = analytical_solution(0.0, grid)
    u_1 = analytical_solution(dt, grid)

    stepper = ex.stepper.Advection(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        velocity=velocity,
    )

    u_1_pred = stepper(u_0)

    assert u_1_pred == pytest.approx(u_1, rel=1e-4)


def test_diffusion_1d():
    num_spatial_dims = 1
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    diffusivity = 0.1

    analytical_solution = lambda t, x: jnp.exp(
        -((4 * 2 * jnp.pi / domain_extent) ** 2) * diffusivity * t
    ) * jnp.sin(4 * 2 * jnp.pi * x / domain_extent)

    grid = ex.get_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = analytical_solution(0.0, grid)
    u_1 = analytical_solution(dt, grid)

    stepper = ex.stepper.Diffusion(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        diffusivity=diffusivity,
    )

    u_1_pred = stepper(u_0)

    assert u_1_pred == pytest.approx(u_1, abs=1e-5)


# Nonlinear steppers

# Burgers can be test by comparing it with the solution obtained by Cole-Hopf
# transformation.


# The Korteveg-de Vries equation has an analytical solution, given the initial
# condition is a soliton.
