"""
Stochastic utilities for Exponax.

Contents
--------
stochastic_rollout(stepper, num_steps, *, include_init)
    Like ``exponax.rollout`` but threads a PRNGKey through a
    stochastic stepper via ``jax.lax.scan``.  Fully JIT-compatible.

stochastic_ensemble_rollout(stepper, num_steps, num_samples, *, include_init)
    ``jax.vmap``-wrapped version of ``stochastic_rollout`` that
    produces M independent trajectories from a single call.
    Returns shape ``(M, T, 1, *N)`` when ``include_init=False``,
    or ``(M, T+1, 1, *N)`` when ``include_init=True``.

structure_factor(ensemble, *, burn_in_fraction)
    Compute ``S(k) = ⟨|û_k|²⟩`` from an ensemble of physical-space
    trajectories (average over the last ``(1 - burn_in_fraction)``
    fraction of time steps).

richardson_weak_extrapolation(stepper_coarse, stepper_fine, u0, *, ...)
    Combine runs at Δt and Δt/2 to boost the *weak* order by one via
    Romberg extrapolation:  E_rich ≈ 2·E_fine - E_coarse.

strang_split_step(spectral_stepper, ssa_step_fn, u, ssa_state, dt, key, *)
    Scaffold for operator-split hybrid SSA / spectral stepping
    (Strang splitting: half-SSA → full-spectral → half-SSA).
    ``ssa_step_fn`` is a user-supplied Python callable invoked via
    ``jax.pure_callback``; it is NOT JIT-compiled.

NOTE on what is NOT provided
-----------------------------
* Richardson *strong* extrapolation: improving the strong order
    requires coupling at the path level (same Wiener increments), which
    exponential Euler-Maruyama does not yet support at order > 1/2.
* Full Milstein for non-gradient noise: iterated Wiener integrals are
    required; not implemented.
* Adaptive Δt: contradicts Exponax's fixed-step design.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from exponax._spectral import get_fourier_coefficients


# ---------------------------------------------------------------------------
# stochastic_rollout
# ---------------------------------------------------------------------------

def stochastic_rollout(
    stepper: eqx.Module,
    num_steps: int,
    *,
    include_init: bool = True,
) -> Callable[[Float[Array, "1 *N"], PRNGKeyArray], Float[Array, "T 1 *N"]]:
    """Return a function that rolls a stochastic stepper forward.

    The returned function has signature::

        trajectory = rollout_fn(u_init, key)

    where ``trajectory`` has shape ``(T+1, 1, *N)`` if
    ``include_init=True`` else ``(T, 1, *N)``.

    The function is fully JIT-compatible because it uses
    ``jax.lax.scan`` internally rather than a Python-level loop.

    **Arguments:**

    - `stepper`: A stochastic stepper whose ``__call__`` accepts
        ``(u, key=key)`` as a keyword-only key argument.
    - `num_steps`: Number of time steps T to take.
    - `include_init`: Whether to prepend the initial condition to the
        returned trajectory array.

    **Example:**

    ```python
    rollout_fn = stochastic_rollout(stepper, 500)
    trajectory = jax.jit(rollout_fn)(u_0, jax.random.PRNGKey(0))
    trajectory.shape  # (501, 1, N)
    ```
    """
    def _rollout(
        u_init: Float[Array, "1 *N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "T 1 *N"]:

        def _step(carry, _):
            u, rng = carry
            rng, subkey = jax.random.split(rng)
            u_next = stepper(u, key=subkey)
            return (u_next, rng), u_next

        (_, _), trajectory = jax.lax.scan(
            _step, (u_init, key), None, length=num_steps
        )

        if include_init:
            trajectory = jnp.concatenate(
                [u_init[jnp.newaxis], trajectory], axis=0
            )

        return trajectory

    return _rollout


# ---------------------------------------------------------------------------
# stochastic_ensemble_rollout
# ---------------------------------------------------------------------------

def stochastic_ensemble_rollout(
    stepper: eqx.Module,
    num_steps: int,
    num_samples: int,
    *,
    include_init: bool = True,
) -> Callable[
    [Float[Array, "1 *N"], PRNGKeyArray],
    Float[Array, "M T 1 *N"],
]:
    """Return a batched rollout function producing M independent paths.

    The returned function has signature::

        ensemble = ensemble_fn(u_init, key)

    where ``ensemble`` has shape ``(M, T+1, 1, *N)``
    (or ``(M, T, 1, *N)`` when ``include_init=False``).

    **Arguments:**

    - `stepper`: Forwarded to ``stochastic_rollout``.
    - `num_steps`: Number of time steps T to take.
    - `num_samples`: Number of Monte-Carlo trajectories M.
    - `include_init`: Whether to prepend the initial condition.

    **Example:**

    ```python
    ens_fn = stochastic_ensemble_rollout(stepper, 200, num_samples=256)
    ensemble = jax.jit(ens_fn)(u_0, jax.random.PRNGKey(0))
    ensemble.shape  # (256, 201, 1, N)
    ```
    """
    single_rollout = stochastic_rollout(
        stepper, num_steps, include_init=include_init
    )

    def _ensemble(
        u_init: Float[Array, "1 *N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "M T 1 *N"]:
        keys = jax.random.split(key, num_samples)
        return jax.vmap(lambda k: single_rollout(u_init, k))(keys)

    return _ensemble


# ---------------------------------------------------------------------------
# structure_factor
# ---------------------------------------------------------------------------

def structure_factor(
    ensemble: Float[Array, "M T 1 *N"],
    *,
    burn_in_fraction: float = 0.5,
) -> Float[Array, "1 *K"]:
    """Compute the empirical structure factor from a trajectory ensemble.

    Returns ``S(k) = ⟨|û_k|²⟩`` where the average is taken over all
    Monte-Carlo samples and over the stationary portion of each
    trajectory (the last ``1 - burn_in_fraction`` of time steps).

    Coefficients are extracted in *coefficient-space* convention (i.e.
    with ``scaling_compensation_mode="coef_extraction"`` and
    ``round=None``), matching the analytical formula

    ```
        C_k = Q_k / (2 ν |k|²_phys)
    ```

    which is the invariant measure of the linear (λ=0) stochastic
    Allen-Cahn equation.

    **Arguments:**

    - `ensemble`: Array of shape ``(M, T, 1, *N)`` where M is the
        number of Monte-Carlo trajectories and T the number of time
        steps (excluding the initial condition when ``include_init=False``
        was used in ``stochastic_ensemble_rollout``).
    - `burn_in_fraction`: Fraction of time steps to discard as burn-in
        before accumulating statistics.  Defaults to ``0.5``.

    **Returns:**

    - ``S_k``: Array of shape ``(1, *K)`` where ``*K`` is the
        wavenumber-space shape of the rfft of the spatial field.
    """
    M, T, *spatial_shape = ensemble.shape
    burn_in = int(T * burn_in_fraction)
    tail = ensemble[:, burn_in:, ...]             # (M, T_stat, 1, *N)
    flat = tail.reshape((-1,) + tuple(spatial_shape))  # (M*T_stat, 1, *N)

    coeffs_per_snapshot = jax.vmap(
        lambda u: get_fourier_coefficients(
            u, scaling_compensation_mode="coef_extraction", round=None
        )
    )(flat)  # (M*T_stat, 1, *K)

    S_k = jnp.mean(jnp.abs(coeffs_per_snapshot) ** 2, axis=0)  # (1, *K)
    return S_k


# ---------------------------------------------------------------------------
# richardson_weak_extrapolation
# ---------------------------------------------------------------------------

def richardson_weak_extrapolation(
    stepper_coarse: eqx.Module,
    stepper_fine: eqx.Module,
    u0: Float[Array, "1 *N"],
    num_steps_coarse: int,
    key: PRNGKeyArray,
    *,
    num_samples: int = 512,
) -> dict:
    """Boost weak order via Romberg/Richardson extrapolation.

    Runs two independent ensembles at the same physical end-time
    ``T = num_steps_coarse * Δt_coarse``:

    - **Coarse**: ``stepper_coarse`` for ``num_steps_coarse`` steps at ``Δt``.
    - **Fine**:   ``stepper_fine``   for ``2 * num_steps_coarse`` steps at
        ``Δt/2`` (same wall-clock time T).

    The Richardson extrapolated mean field is::

        E_rich[u(T)] ≈ 2·E_fine[u(T)] - E_coarse[u(T)]

    which doubles the weak convergence order (e.g. order 1 → order 2
    for additive noise with Q-Wiener forcing).

    **Arguments:**

    - `stepper_coarse`: Stepper at time step Δt.
    - `stepper_fine`: Stepper at time step Δt/2 (same physical
        parameters, half the ``dt``).
    - `u0`: Initial condition, shape ``(1, *N)``.
    - `num_steps_coarse`: Number of coarse steps T.
    - `key`: Master PRNGKey; split internally for the two ensembles.
    - `num_samples`: Monte-Carlo sample size M for both ensembles.

    **Returns:**

    A ``dict`` with keys:

    - ``"mean_coarse"``, ``"mean_fine"``, ``"mean_rich"``: Ensemble-mean
        final states.
    - ``"var_coarse"``, ``"var_fine"``: Ensemble variance of the final
        state.
    """
    key_c, key_f = jax.random.split(key)

    ens_coarse_fn = stochastic_ensemble_rollout(
        stepper_coarse, num_steps_coarse, num_samples, include_init=False
    )
    ens_fine_fn = stochastic_ensemble_rollout(
        stepper_fine, 2 * num_steps_coarse, num_samples, include_init=False
    )

    ens_c = jax.jit(ens_coarse_fn)(u0, key_c)   # (M, T, 1, *N)
    ens_f = jax.jit(ens_fine_fn)(u0, key_f)

    final_c = ens_c[:, -1, ...]   # (M, 1, *N)
    final_f = ens_f[:, -1, ...]

    mean_c = jnp.mean(final_c, axis=0)
    mean_f = jnp.mean(final_f, axis=0)
    mean_rich = 2.0 * mean_f - mean_c

    var_c = jnp.var(final_c, axis=0)
    var_f = jnp.var(final_f, axis=0)

    return {
        "mean_coarse": mean_c,
        "mean_fine": mean_f,
        "mean_rich": mean_rich,
        "var_coarse": var_c,
        "var_fine": var_f,
    }


# ---------------------------------------------------------------------------
# strang_split_step  (hybrid SSA scaffold — NOT JIT-compiled)
# ---------------------------------------------------------------------------

def strang_split_step(
    spectral_stepper: eqx.Module,
    ssa_step_fn: Callable,
    u: Float[Array, "1 *N"],
    ssa_state: dict,
    dt: float,
    key: PRNGKeyArray,
    *,
    domain_extent: float,
    num_points: int,
    mollifier_cutoff: float = 0.5,
) -> tuple[Float[Array, "1 *N"], dict]:
    """One Strang-split hybrid spectral / SSA step.

    Implements the pattern: half-SSA(Δt/2) → full-spectral(Δt) →
    half-SSA(Δt/2), which is second-order accurate in Δt for operator
    splitting (Strang, 1968).

    This is a scaffold — ``ssa_step_fn`` is a user-supplied Python
    callable and is NOT JIT-compiled.  The spectral step is pure JAX
    and benefits from JIT if called inside a compiled context.

    **Arguments:**

    - `spectral_stepper`: A ``StochasticAllenCahn`` instance.
    - `ssa_step_fn`: Callable with signature::

            new_ssa_state = ssa_step_fn(ssa_state, dt_half)

        Must update the SSA compartment counts and return the updated
        state dict including a ``"delta_concentration"`` field
        (numpy array of shape ``(1, *N)``) representing the concentration
        change to inject into the PDE field.
    - `u`: Current PDE field, shape ``(1, *N)``.
    - `ssa_state`: Current SSA state dictionary (Python-level, not JAX).
    - `dt`: Full time step; the SSA steps use ``dt/2`` each.
    - `key`: PRNGKey for the spectral noise increment.
    - `domain_extent`: Domain side-length L (used for normalisation).
    - `num_points`: Grid points N per dimension.
    - `mollifier_cutoff`: Fraction of Fourier modes retained by the
        band-limited mollifier that converts discrete SSA outcomes to a
        smooth field perturbation.  Default ``0.5`` retains the lowest
        half of resolved modes.

    **Returns:**

    - ``(u_new, ssa_state_new)``: Updated PDE field and SSA state.

    **Notes:**

    Gibbs artefacts from injecting discrete SSA outcomes into the spectral
    field are suppressed by a sharp Fourier truncation mollifier.  A
    smoother Gaussian window is straightforward to substitute.

    **References:**

    - Harrison, J. U., & Yates, C. A. (2016). 
        The two-regime method for optimizing stochastic 
        reaction-diffusion simulations. 
        *Journal of The Royal Society Interface*,
        9(70), 859-868. https://doi.org/10.1098/rsif.2011.0574
        (Strang-split PDE/SSA coupling with band-limited mollifier.)

    - Strang, G. (1968). On the construction and comparison of difference
        schemes. *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
        https://doi.org/10.1137/0705041 (Second-order operator splitting.)

    - Gillespie, D. T. (1977). Exact stochastic simulation of coupled
        chemical reactions. *The Journal of Physical Chemistry*, 81(25),
        2340-2361. https://doi.org/10.1021/j100540a008 
        (Stochastic simulation algorithm.)
    """
    num_spatial_dims = u.ndim - 1
    spatial_axes = tuple(range(-num_spatial_dims, 0))
    N_shape = (num_points,) * num_spatial_dims

    def _mollify(delta: Float[Array, "1 *N"]) -> Float[Array, "1 *N"]:
        """Band-limit a physical-space perturbation with a sharp Fourier cut."""
        delta_hat = jnp.fft.rfftn(delta, axes=spatial_axes)
        rfft_n = num_points // 2 + 1
        max_k = int(mollifier_cutoff * (num_points // 2))
        mask_1d = jnp.arange(rfft_n) <= max_k
        if num_spatial_dims == 1:
            mask = mask_1d[jnp.newaxis, :]
        else:  # 2D
            mask_full = (jnp.arange(num_points) <= max_k)
            mask_full = mask_full[:, jnp.newaxis] & mask_1d[jnp.newaxis, :]
            mask = mask_full[jnp.newaxis, :]
        return jnp.fft.irfftn(delta_hat * mask, s=N_shape, axes=spatial_axes).real

    # ── Half SSA step ────────────────────────────────────────────────
    ssa_state_mid = ssa_step_fn(ssa_state, dt / 2.0)
    delta_mid = jnp.array(ssa_state_mid.get("delta_concentration", 0.0))
    u = u + _mollify(delta_mid)

    # ── Full spectral step ───────────────────────────────────────────
    key, subkey = jax.random.split(key)
    u = spectral_stepper(u, key=subkey)

    # ── Half SSA step ────────────────────────────────────────────────
    ssa_state_new = ssa_step_fn(ssa_state_mid, dt / 2.0)
    delta_end = jnp.array(ssa_state_new.get("delta_concentration", 0.0))
    u = u + _mollify(delta_end)

    return u, ssa_state_new
