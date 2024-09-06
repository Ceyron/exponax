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

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
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


def test_advection_2d():
    num_spatial_dims = 2
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    velocity = jnp.array([0.1, 0.2])

    analytical_solution = lambda t, x: jnp.sin(
        4 * 2 * jnp.pi * (x[0:1] - velocity[0] * t) / domain_extent
    ) * jnp.sin(6 * 2 * jnp.pi * (x[1:2] - velocity[1] * t) / domain_extent)

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
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

    # Primarily only check the absolute difference
    assert u_1_pred == pytest.approx(u_1, rel=1e-3)


def test_advection_3d():
    num_spatial_dims = 3
    domain_extent = 10.0
    num_points = 40
    dt = 0.1
    velocity = jnp.array([0.1, 0.2, 0.3])

    analytical_solution = (
        lambda t, x: jnp.sin(
            4 * 2 * jnp.pi * (x[0:1] - velocity[0] * t) / domain_extent
        )
        * jnp.sin(6 * 2 * jnp.pi * (x[1:2] - velocity[1] * t) / domain_extent)
        * jnp.sin(8 * 2 * jnp.pi * (x[2:3] - velocity[2] * t) / domain_extent)
    )

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
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

    # Primarily only check the absolute difference
    assert u_1_pred == pytest.approx(u_1, rel=1e-3)


def test_diffusion_1d():
    num_spatial_dims = 1
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    diffusivity = 0.1

    analytical_solution = lambda t, x: jnp.exp(
        -((4 * 2 * jnp.pi / domain_extent) ** 2) * diffusivity * t
    ) * jnp.sin(4 * 2 * jnp.pi * x / domain_extent)

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
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


def test_diffusion_2d():
    num_spatial_dims = 2
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    diffusivity = 0.1

    def analytical_solution(t, x):
        # Third sine mode in x-direction and fourth sine mode in y-direction
        third_sine_mode_x = jnp.sin(3 * 2 * jnp.pi * x[0:1] / domain_extent)
        fourth_sine_mode_y = jnp.sin(4 * 2 * jnp.pi * x[1:2] / domain_extent)
        exponent = (
            -diffusivity
            * (
                (3 * 2 * jnp.pi / domain_extent) ** 2
                + (4 * 2 * jnp.pi / domain_extent) ** 2
            )
            * t
        )
        exp_term = jnp.exp(exponent)
        return exp_term * third_sine_mode_x * fourth_sine_mode_y

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)

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


def test_diffusion_3d():
    num_spatial_dims = 3
    domain_extent = 10.0
    num_points = 40
    dt = 0.1
    diffusivity = 0.1

    def analytical_solution(t, x):
        # Third sine mode in x-direction, fourth sine mode in y-direction, and
        # fifth sine mode in z-direction
        third_sine_mode_x = jnp.sin(3 * 2 * jnp.pi * x[0:1] / domain_extent)
        fourth_sine_mode_y = jnp.sin(4 * 2 * jnp.pi * x[1:2] / domain_extent)
        fifth_sine_mode_z = jnp.sin(5 * 2 * jnp.pi * x[2:3] / domain_extent)
        exponent = (
            -diffusivity
            * (
                (3 * 2 * jnp.pi / domain_extent) ** 2
                + (4 * 2 * jnp.pi / domain_extent) ** 2
                + (5 * 2 * jnp.pi / domain_extent) ** 2
            )
            * t
        )
        exp_term = jnp.exp(exponent)
        return exp_term * third_sine_mode_x * fourth_sine_mode_y * fifth_sine_mode_z

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)

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


def test_validation_poisson_1d():
    DOMAIN_EXTENT = 1.0
    NUM_POINTS = 50

    grid = ex.make_grid(1, DOMAIN_EXTENT, NUM_POINTS)

    rhs = jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid)
    analytical_solution = (
        jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid) / (2 * jnp.pi / DOMAIN_EXTENT) ** 2
    )

    poisson_solver = ex.poisson.Poisson(
        1,
        DOMAIN_EXTENT,
        NUM_POINTS,
    )

    u = poisson_solver(rhs)

    assert u == pytest.approx(analytical_solution, abs=1e-6)


def test_validation_poisson_2d():
    DOMAIN_EXTENT = 3.0
    NUM_POINTS = 25

    grid = ex.make_grid(2, DOMAIN_EXTENT, NUM_POINTS)

    rhs = jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[0:1]) * jnp.sin(
        2 * jnp.pi / DOMAIN_EXTENT * grid[1:2]
    )
    analytical_solution = (
        1
        / 2
        * jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[0:1])
        * jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[1:2])
        / (2 * jnp.pi / DOMAIN_EXTENT) ** 2
    )

    poisson_solver = ex.poisson.Poisson(
        2,
        DOMAIN_EXTENT,
        NUM_POINTS,
    )

    u = poisson_solver(rhs)

    assert u == pytest.approx(analytical_solution, abs=1e-6)


def test_validation_poisson_3d():
    DOMAIN_EXTENT = 3.0
    NUM_POINTS = 15

    grid = ex.make_grid(3, DOMAIN_EXTENT, NUM_POINTS)

    rhs = (
        jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[0:1])
        * jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[1:2])
        * jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[2:3])
    )
    analytical_solution = (
        1
        / 3
        * jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[0:1])
        * jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[1:2])
        * jnp.sin(2 * jnp.pi / DOMAIN_EXTENT * grid[2:3])
        / (2 * jnp.pi / DOMAIN_EXTENT) ** 2
    )

    poisson_solver = ex.poisson.Poisson(
        3,
        DOMAIN_EXTENT,
        NUM_POINTS,
    )

    u = poisson_solver(rhs)

    assert u == pytest.approx(analytical_solution, abs=1e-6)


# Find more validations that do not fit the format of a unit test in the
# `validation/` directory. For example, the following validations:
