import jax
import jax.numpy as jnp
import pytest

import exponax as ex


def test_repeated_stepper():
    DOMAIN_EXTENT = 1.0
    NUM_POINTS = 81
    DT = 0.1
    NUM_REPEATS = 10

    burgers_stepper = ex.stepper.Burgers(1, DOMAIN_EXTENT, NUM_POINTS, DT)

    burgers_stepper_repeated = ex.RepeatedStepper(burgers_stepper, NUM_REPEATS)

    burgers_stepper_repeated_manually = ex.repeat(burgers_stepper, NUM_REPEATS)

    u_0 = ex.ic.RandomTruncatedFourierSeries(1, max_one=True)(
        NUM_POINTS, key=jax.random.PRNGKey(0)
    )

    u_final = burgers_stepper_repeated(u_0)
    u_final_manually = burgers_stepper_repeated_manually(u_0)

    # Need a looser rel tolerance because Burgers is a decaying phenomenon,
    # hence the expected/reference state has low magnitude after 10 steps.
    assert u_final == pytest.approx(u_final_manually, rel=1e-3)


def test_repeated_stepper_dt():
    """RepeatedStepper.dt should equal sub_stepper.dt * num_sub_steps."""
    DT = 0.05
    NUM_REPEATS = 8

    sub_stepper = ex.stepper.Diffusion(1, 3.0, 32, DT, diffusivity=0.1)
    repeated = ex.RepeatedStepper(sub_stepper, NUM_REPEATS)

    assert repeated.dt == pytest.approx(DT * NUM_REPEATS, abs=1e-10)


def test_repeated_stepper_convergence():
    """
    For Burgers 1D, sub-stepping with many small steps should be more
    accurate than a single large step.
    """
    domain_extent = 3.0
    num_points = 64
    total_dt = 0.2
    diffusivity = 0.05

    grid = ex.make_grid(1, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent)

    # Reference: very fine sub-stepping
    dt_ref = total_dt / 64
    ref_stepper = ex.stepper.Burgers(
        1, domain_extent, num_points, dt_ref, diffusivity=diffusivity
    )
    u_ref = ex.repeat(ref_stepper, 64)(u_0)

    # Single large step
    single_stepper = ex.stepper.Burgers(
        1, domain_extent, num_points, total_dt, diffusivity=diffusivity
    )
    u_single = single_stepper(u_0)

    # Sub-stepped: 8 steps of total_dt/8
    dt_sub = total_dt / 8
    sub_stepper = ex.stepper.Burgers(
        1, domain_extent, num_points, dt_sub, diffusivity=diffusivity
    )
    repeated = ex.RepeatedStepper(sub_stepper, 8)
    u_sub = repeated(u_0)

    error_single = float(jnp.sqrt(jnp.mean((u_single - u_ref) ** 2)))
    error_sub = float(jnp.sqrt(jnp.mean((u_sub - u_ref) ** 2)))

    # Sub-stepped should be more accurate
    assert error_sub < error_single
