import jax.numpy as jnp
import pytest

import exponax as ex


def test_forced_stepper_zero_forcing():
    """With zero forcing, ForcedStepper should match the unforced stepper."""
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 64
    dt = 0.1

    stepper = ex.stepper.Diffusion(
        num_spatial_dims, domain_extent, num_points, dt, diffusivity=0.1
    )
    forced_stepper = ex.ForcedStepper(stepper)

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent)
    f = jnp.zeros_like(u_0)

    u_unforced = stepper(u_0)
    u_forced = forced_stepper(u_0, f)

    assert u_forced == pytest.approx(u_unforced, abs=1e-6)


def test_forced_stepper_constant_forcing():
    """
    ForcedStepper uses forward Euler splitting: u_forced = stepper(u + dt*f).
    Verify this directly.
    """
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 64
    dt = 0.1

    stepper = ex.stepper.Diffusion(
        num_spatial_dims, domain_extent, num_points, dt, diffusivity=0.1
    )
    forced_stepper = ex.ForcedStepper(stepper)

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent)
    f = jnp.ones_like(u_0) * 0.5

    u_forced = forced_stepper(u_0, f)

    # The forced stepper should be equivalent to: stepper(u + dt*f)
    u_expected = stepper(u_0 + dt * f)

    assert u_forced == pytest.approx(u_expected, abs=1e-6)


def test_forced_stepper_energy_injection():
    """
    For a purely dissipative PDE (diffusion), forcing should result in
    higher energy than the unforced case.
    """
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 64
    dt = 0.1

    stepper = ex.stepper.Diffusion(
        num_spatial_dims, domain_extent, num_points, dt, diffusivity=0.1
    )
    forced_stepper = ex.ForcedStepper(stepper)

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent)
    f = jnp.sin(4 * jnp.pi * grid / domain_extent) * 2.0

    u_unforced = stepper(u_0)
    u_forced = forced_stepper(u_0, f)

    energy_unforced = jnp.mean(u_unforced**2)
    energy_forced = jnp.mean(u_forced**2)

    assert energy_forced > energy_unforced
