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


def test_rollout_length_with_init():
    """rollout with include_init=True should return n+1 states."""
    stepper = ex.stepper.Diffusion(1, 3.0, 32, 0.1, diffusivity=0.1)
    grid = ex.make_grid(1, 3.0, 32)
    u_0 = jnp.sin(2 * jnp.pi * grid / 3.0)

    n = 5
    trj = ex.rollout(stepper, n, include_init=True)(u_0)

    assert trj.shape[0] == n + 1


def test_rollout_length_without_init():
    """rollout with include_init=False should return n states."""
    stepper = ex.stepper.Diffusion(1, 3.0, 32, 0.1, diffusivity=0.1)
    grid = ex.make_grid(1, 3.0, 32)
    u_0 = jnp.sin(2 * jnp.pi * grid / 3.0)

    n = 5
    trj = ex.rollout(stepper, n, include_init=False)(u_0)

    assert trj.shape[0] == n


def test_rollout_matches_manual_loop():
    """rollout output should match manually stepping in a Python loop."""
    stepper = ex.stepper.Diffusion(1, 3.0, 32, 0.1, diffusivity=0.1)
    grid = ex.make_grid(1, 3.0, 32)
    u_0 = jnp.sin(2 * jnp.pi * grid / 3.0)

    n = 3
    trj = ex.rollout(stepper, n, include_init=True)(u_0)

    # Manual loop
    u = u_0
    manual_states = [u]
    for _ in range(n):
        u = stepper(u)
        manual_states.append(u)

    for i in range(n + 1):
        assert trj[i] == pytest.approx(manual_states[i], abs=1e-6)


def test_repeat_matches_rollout_final():
    """repeat(stepper, n)(u0) should match rollout(stepper, n)(u0)[-1]."""
    stepper = ex.stepper.Diffusion(1, 3.0, 32, 0.1, diffusivity=0.1)
    grid = ex.make_grid(1, 3.0, 32)
    u_0 = jnp.sin(2 * jnp.pi * grid / 3.0)

    n = 5
    trj = ex.rollout(stepper, n, include_init=False)(u_0)
    u_final_rollout = trj[-1]

    u_final_repeat = ex.repeat(stepper, n)(u_0)

    assert u_final_repeat == pytest.approx(u_final_rollout, abs=1e-6)
