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


# ===========================================================================
# make_grid tests
# ===========================================================================


class TestMakeGrid:
    @pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
    def test_zero_centered(self, num_spatial_dims):
        """Zero-centered grid should range from -L/2 to ~L/2."""
        L = 4.0
        N = 16
        grid = ex.make_grid(num_spatial_dims, L, N, zero_centered=True)
        # First point should be at -L/2
        assert float(grid[0].flatten()[0]) == pytest.approx(-L / 2, abs=1e-6)
        # Last point should be at L/2 - L/N
        last_expected = L / 2 - L / N
        assert float(grid[0].flatten()[-1]) == pytest.approx(last_expected, abs=1e-6)

    def test_zero_centered_vs_not(self):
        """Zero-centered grid should be shifted by -L/2 compared to regular."""
        L = 2.0
        N = 32
        grid = ex.make_grid(1, L, N)
        grid_zc = ex.make_grid(1, L, N, zero_centered=True)
        assert grid_zc == pytest.approx(grid - L / 2, abs=1e-6)

    def test_full_grid_1d(self):
        """Full grid should have N+1 points."""
        grid = ex.make_grid(1, 1.0, 10, full=True)
        assert grid.shape == (1, 11)

    def test_full_grid_2d(self):
        grid = ex.make_grid(2, 1.0, 10, full=True)
        assert grid.shape == (2, 11, 11)

    def test_zero_centered_full(self):
        """Zero-centered full grid: first at -L/2, last at L/2."""
        L = 4.0
        N = 16
        grid = ex.make_grid(1, L, N, full=True, zero_centered=True)
        assert float(grid[0, 0]) == pytest.approx(-L / 2, abs=1e-6)
        assert float(grid[0, -1]) == pytest.approx(L / 2, abs=1e-6)


# ===========================================================================
# rollout with aux tests
# ===========================================================================


class TestRolloutWithAux:
    def test_constant_aux(self):
        """rollout with takes_aux=True and constant_aux=True."""

        def stepper_fn(u, aux):
            return u + aux

        n = 5
        rollout_fn = ex.rollout(stepper_fn, n, takes_aux=True, constant_aux=True)
        u_0 = jnp.ones((1, 32))
        aux = 0.1 * jnp.ones((1, 32))

        trj = rollout_fn(u_0, aux)
        assert trj.shape == (n, 1, 32)
        # After 5 steps: u_0 + 5*aux = 1.0 + 0.5 = 1.5
        assert trj[-1] == pytest.approx(u_0 + n * aux, abs=1e-6)

    def test_variable_aux(self):
        """rollout with takes_aux=True and constant_aux=False."""

        def stepper_fn(u, aux):
            return u + aux

        n = 3
        rollout_fn = ex.rollout(stepper_fn, n, takes_aux=True, constant_aux=False)
        u_0 = jnp.zeros((1, 32))
        # Variable aux: different value at each step
        aux = jnp.stack(
            [
                0.1 * jnp.ones((1, 32)),
                0.2 * jnp.ones((1, 32)),
                0.3 * jnp.ones((1, 32)),
            ]
        )

        trj = rollout_fn(u_0, aux)
        assert trj.shape == (n, 1, 32)
        # After 3 steps: 0 + 0.1 + 0.2 + 0.3 = 0.6
        assert trj[-1] == pytest.approx(0.6 * jnp.ones((1, 32)), abs=1e-5)

    def test_rollout_with_aux_include_init(self):
        """rollout with takes_aux and include_init should return n+1 states."""

        def stepper_fn(u, aux):
            return u + aux

        n = 4
        rollout_fn = ex.rollout(
            stepper_fn, n, takes_aux=True, constant_aux=True, include_init=True
        )
        u_0 = jnp.ones((1, 16))
        aux = 0.5 * jnp.ones((1, 16))

        trj = rollout_fn(u_0, aux)
        assert trj.shape == (n + 1, 1, 16)
        assert trj[0] == pytest.approx(u_0, abs=1e-6)


# ===========================================================================
# repeat with aux tests
# ===========================================================================


class TestRepeatWithAux:
    def test_constant_aux(self):
        """repeat with takes_aux=True and constant_aux=True."""

        def stepper_fn(u, aux):
            return u + aux

        n = 5
        repeat_fn = ex.repeat(stepper_fn, n, takes_aux=True, constant_aux=True)
        u_0 = jnp.ones((1, 32))
        aux = 0.1 * jnp.ones((1, 32))

        u_final = repeat_fn(u_0, aux)
        assert u_final.shape == (1, 32)
        assert u_final == pytest.approx(u_0 + n * aux, abs=1e-6)

    def test_variable_aux(self):
        """repeat with takes_aux=True and constant_aux=False."""

        def stepper_fn(u, aux):
            return u + aux

        n = 3
        repeat_fn = ex.repeat(stepper_fn, n, takes_aux=True, constant_aux=False)
        u_0 = jnp.zeros((1, 32))
        aux = jnp.stack(
            [
                0.1 * jnp.ones((1, 32)),
                0.2 * jnp.ones((1, 32)),
                0.3 * jnp.ones((1, 32)),
            ]
        )

        u_final = repeat_fn(u_0, aux)
        assert u_final.shape == (1, 32)
        assert u_final == pytest.approx(0.6 * jnp.ones((1, 32)), abs=1e-5)

    def test_repeat_with_aux_matches_rollout(self):
        """repeat with aux should match the final state from rollout."""

        def stepper_fn(u, aux):
            return u * 0.9 + aux

        n = 5
        u_0 = jnp.ones((1, 16))
        aux = 0.05 * jnp.ones((1, 16))

        trj = ex.rollout(stepper_fn, n, takes_aux=True, constant_aux=True)(u_0, aux)
        u_final = ex.repeat(stepper_fn, n, takes_aux=True, constant_aux=True)(u_0, aux)

        assert u_final == pytest.approx(trj[-1], abs=1e-6)


# ===========================================================================
# stack_sub_trajectories tests
# ===========================================================================


class TestStackSubTrajectories:
    def test_basic(self):
        trj = jnp.arange(10).reshape(10, 1)
        sub_trjs = ex.stack_sub_trajectories(trj, sub_len=3)
        # n_stacks = 10 - 3 + 1 = 8
        assert sub_trjs.shape == (8, 3, 1)
        # First sub-trajectory should be [0, 1, 2]
        assert sub_trjs[0, :, 0] == pytest.approx(jnp.array([0, 1, 2]))
        # Last sub-trajectory should be [7, 8, 9]
        assert sub_trjs[-1, :, 0] == pytest.approx(jnp.array([7, 8, 9]))

    def test_sub_len_equals_length(self):
        trj = jnp.arange(5).reshape(5, 1)
        sub_trjs = ex.stack_sub_trajectories(trj, sub_len=5)
        assert sub_trjs.shape == (1, 5, 1)

    def test_sub_len_too_large_raises(self):
        trj = jnp.arange(5).reshape(5, 1)
        with pytest.raises(ValueError, match="smaller"):
            ex.stack_sub_trajectories(trj, sub_len=6)
