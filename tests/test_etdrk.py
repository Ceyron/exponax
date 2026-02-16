import jax.numpy as jnp
import pytest

import exponax as ex
from exponax.etdrk._utils import roots_of_unity


def test_roots_of_unity():
    """Verify roots_of_unity returns M points on the unit circle."""
    M = 16
    roots = roots_of_unity(M)

    assert roots.shape == (M,)
    # All roots should lie on the unit circle
    assert jnp.abs(roots) == pytest.approx(jnp.ones(M), abs=1e-6)
    # Roots should sum to approximately zero (equally spaced on circle)
    assert jnp.sum(roots) == pytest.approx(0.0, abs=1e-6)


def test_etdrk0_exact_for_linear():
    """ETDRK0 (used for purely linear PDEs) should solve diffusion exactly."""
    num_spatial_dims = 1
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    diffusivity = 0.1

    def analytical_solution(t, x):
        return jnp.exp(
            -((4 * 2 * jnp.pi / domain_extent) ** 2) * diffusivity * t
        ) * jnp.sin(4 * 2 * jnp.pi * x / domain_extent)

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = analytical_solution(0.0, grid)
    u_1_exact = analytical_solution(dt, grid)

    # Diffusion stepper uses order=0 (ETDRK0) by default
    stepper = ex.stepper.Diffusion(
        num_spatial_dims, domain_extent, num_points, dt, diffusivity=diffusivity
    )

    u_1_pred = stepper(u_0)

    # Should be exact to machine precision (linear PDE, spectral in space)
    assert u_1_pred == pytest.approx(u_1_exact, abs=1e-5)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_etdrk_convergence_order(order):
    """
    Verify that ETDRK of the given order converges at rate O(dt^order) on
    Burgers equation. We use large enough dt values that the temporal error
    dominates over float32 precision.
    """
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 64
    diffusivity = 0.02

    # Smooth initial condition with moderate amplitude
    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent)

    # Use large dt so temporal errors dominate, scale with order so
    # higher-order methods don't converge "too fast" to the noise floor
    dt_base = 0.2

    # Reference solution: very small dt with ETDRK4 (highest order)
    n_ref_steps = 256
    dt_ref = dt_base / n_ref_steps
    stepper_ref = ex.stepper.Burgers(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt_ref,
        diffusivity=diffusivity,
        order=4,
    )
    u_ref = ex.repeat(stepper_ref, n_ref_steps)(u_0)

    # Test with refinements: 1, 2, 4 sub-steps to cover the same T=dt_base
    errors = []
    for refinement in [1, 2, 4]:
        n_steps = refinement
        dt = dt_base / refinement
        stepper = ex.stepper.Burgers(
            num_spatial_dims,
            domain_extent,
            num_points,
            dt,
            diffusivity=diffusivity,
            order=order,
        )
        u_pred = ex.repeat(stepper, n_steps)(u_0)
        error = float(jnp.sqrt(jnp.mean((u_pred - u_ref) ** 2)))
        errors.append(error)

    # Compute convergence rate from first two refinements
    rate = jnp.log2(errors[0] / errors[1])

    # The convergence rate should be approximately equal to the ETDRK order.
    # Allow generous tolerance since we're comparing against a numerical
    # reference and using single precision.
    assert rate > order - 0.25, (
        f"ETDRK{order}: expected rate ~{order}, got {rate:.2f} (errors: {errors})"
    )
