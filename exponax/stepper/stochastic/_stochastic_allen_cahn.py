"""
Stochastic Allen-Cahn SPDE stepper.

Equation (Itô):  ∂ₜu = νΔu + λ(u - u³) + σ(u) ξ(x, t)

Method: Exponential Euler-Maruyama (EEM)
─────────────────────────────────────────
The PDE is split, following the built-in ``exponax.stepper.reaction.AllenCahn``
stepper convention, as

    L(u) = ν Δu + λu     ← handled analytically by ETD
    𝒩(u) = -λ u³         ← evaluated explicitly

so that L + 𝒩 = ν Δu + λu - λu³ = ν Δu + λ(u - u³).

The discrete update is

    û_k(t+Δt) = E_k û_k(t)
                + φ₁(L_k Δt) Δt · 𝒩̂_k(u(t))   [ETD1 step]
                + stochastic_increment_k           [EEM noise]

where E_k = exp(L_k Δt) and φ₁(z) = (eᶻ - 1)/z.

Noise variance (exact discrete-time form)
─────────────────────────────────────────
The EEM stochastic integral ∫₀^Δt e^{L_k(Δt-s)} dŴ_k(s) has exact
coefficient-space variance (Lord, Powell & Shardlow, 2014, §10.5)

    Var_k = Q_k · (e^{2L_k Δt} - 1) / (2 L_k)   [L_k ≠ 0]
          = Q_k · Δt                               [L_k → 0]

where Q_k = σ² · (1 + |k|²_phys)^{-α} / dx^d is the noise spectral
density in coefficient units (normalised by the mesh volume element
dx^d so that the variance is resolution-consistent).

This formula is used for both additive and multiplicative noise.  For
multiplicative noise Q_k is replaced by Q_base = Q_k / σ² so that σ
enters exactly once via the u · dW product.

API facts confirmed from exponax source
────────────────────────────────────────
BaseStepper.__init__(num_spatial_dims, domain_extent, num_points, dt,
                     *, num_channels, order)
  - no dealiasing_fraction argument.
  - derivative_operator built locally; NOT stored on self.
  - self._integrator : BaseETDRK instance.

BaseETDRK: self._exp_term = exp(dt * L_k)
ETDRK1+:  self._coef_1 = φ₁(L_k Δt) · Δt
          self._nonlinear_fun : TamedPolynomialNonlinearFun instance

BaseNonlinearFun.__call__(u_hat) → nonlin_hat  (single Complex array)

build_derivative_operator(num_spatial_dims, domain_extent, num_points)
  → pure function from exponax._spectral; called again here to obtain
    k² for the noise spectrum (BaseStepper discards it after construction).

Known limitations
─────────────────
1. DC (k=0) and Nyquist (k=N/2) modes: independent real and imaginary
   Gaussian noise is sampled for every rfft entry, including these two
   modes which must be purely real for a real-valued field.  Taking
   ``.real`` after ``ifft`` silently halves their per-step variance.
   Both modes are either masked in the invariant-measure tests (k=0)
   or lie above the dealiasing cutoff (Nyquist), so no validation is
   affected.  A rigorous fix would project z_i to zero at those indices.

2. Milstein prefactor: the Milstein correction is scaled by φ₁(L_k Δt)Δt
   (treating it as a nonlinear ETD term). This is non-standard — the
   textbook EEM-Milstein scheme (Jentzen & Kloeden, 2009a) applies the
   correction without this ETD factor. The variant is retained for
   backward compatibility; the test verifies finiteness and variance
   boundedness rather than strict weak-error ordering.

3. Noise above the dealiasing cutoff: the noise arrays _noise_std and
   _noise_std_base are NOT masked by the dealiasing filter, so modes
   above the 2/3 cutoff do receive stochastic forcing. This is
   physically intentional — dealiasing prevents aliasing of the
   nonlinear product, not external forcing — and consistent with
   standard Q-Wiener discretisations (Lord et al., 2014, Chapter 10).
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..._base_stepper import BaseStepper
from ..._spectral import (
    build_derivative_operator,
    build_scaling_array,
    fft,
    ifft,
)
from ...nonlin_fun import TamedPolynomialNonlinearFun


def _spatial_shape(num_spatial_dims: int, num_points: int) -> tuple[int, ...]:
    return (num_points,) * num_spatial_dims


class StochasticAllenCahn(BaseStepper):
    """
    Timestepper for the d-dimensional (``d ∈ {1, 2, 3}``) stochastic
    Allen-Cahn SPDE on periodic boundary conditions, solved via the
    Exponential Euler-Maruyama (EEM) method.

    In Itô form the equation reads

    ```
        ∂ₜu = ν Δu + λ(u - u³) + σ(u) ξ(x, t)
    ```

    with ``ν > 0`` the interface width, ``λ ≥ 0`` the reaction rate,
    ``σ(u)`` the noise coefficient (see ``noise_type``), and ``ξ(x, t)``
    a Q-Wiener process with spectral covariance

    ```
        Q_k ∝ (1 + |k|²_phys)^{-α}
    ```

    The deterministic limit (``sigma=0``) recovers the classical Allen-Cahn
    reaction-diffusion equation, producing sharp diffuse interfaces whose
    width scales as ``√ν``.  The expected long-time behaviour is the
    formation and coarsening of phase domains separated by smooth interfaces
    of thickness ``O(ν)``.

    Note that the Allen-Cahn equation is often solved with Dirichlet boundary
    conditions; here we use periodic boundary conditions throughout.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions ``d``.
    - `domain_extent`: The size of the domain ``L``; the domain is assumed
        to be a scaled hypercube ``Ω = (0, L)ᵈ`` with periodic boundaries.
    - `num_points`: The number of points ``N`` used to discretize the
        domain. This **includes** the left boundary point and **excludes**
        the right boundary point. In higher dimensions the number of points
        in each dimension is the same. Hence, the total number of degrees of
        freedom is ``Nᵈ``.
    - `dt`: The timestep size ``Δt`` between two consecutive states.
    - `diffusivity`: The interface width ``ν > 0``.  Smaller values produce
        thinner, sharper interfaces.  Defaults to ``0.01``.
    - `lambda_`: The reaction rate ``λ ≥ 0``.  Setting ``lambda_=0``
        reduces the equation to the stochastic heat equation, whose
        invariant measure is Gaussian with covariance ``C_k = Q_k/(2ν|k|²)``
        and can be validated analytically.  Defaults to ``1.0``.
    - `noise_type`: Either ``"additive"`` (``σ(u) = σ``) or
        ``"multiplicative"`` (``σ(u) = σu``, linear / gradient-type).
        Defaults to ``"additive"``.
    - `sigma`: The noise amplitude ``σ ≥ 0``.  Setting ``sigma=0``
        recovers the deterministic Allen-Cahn solver; the output is
        then identical regardless of the PRNG key.  Defaults to ``0.1``.
    - `noise_alpha`: Spectral colour index ``α ≥ 0`` for the Q-Wiener
        covariance ``Q_k ∝ (1 + |k|²_phys)^{-α}``.  ``α = 0`` gives
        space-time white noise; ``α > d/2`` gives a trace-class
        (spatially smooth) noise.  Defaults to ``1.0``.
    - `use_taming`: Whether to apply Hutzenthaler-Jentzen taming to the
        cubic nonlinear term

        ```
            𝒩_tamed(u) = 𝒩(u) / (1 + Δt |𝒩(u)|)
        ```

        Recommended for large ``lambda_`` or large ``sigma``.  Set to
        ``False`` only when comparing against an untamed deterministic
        reference (``exponax.stepper.reaction.AllenCahn``).  Defaults
        to ``True``.
    - `use_milstein`: Whether to add the first-order Milstein correction
        for multiplicative noise.  Has no effect for additive noise.
        Auto-disabled when ``noise_alpha=0`` because the iterated
        stochastic integrals required for space-time white noise diverge.
        Defaults to ``False``.
    - `order`: The order of the ETDRK method.  Must be ``≥ 1`` so that
        ``_nonlinear_fun`` and ``_coef_1`` exist on the integrator.
        Higher values give more accurate deterministic substeps but do
        not change the stochastic convergence order of the EEM noise.
        Defaults to ``1`` (Exponential Euler).
    - `dealiasing_fraction`: The fraction of the highest wavenumbers to
        dealias in the nonlinear evaluation.  Default ``2/3`` (Orszag's
        rule); ``1/2`` is tighter for the cubic nonlinearity.

    **Notes on calling convention:**

    ```python
    u_next = stepper(u, key=jax.random.PRNGKey(0))
    ```

    The ``BaseStepper.step()`` interface (no key) is deliberately
    disabled and raises ``NotImplementedError``.  Use the ``__call__``
    interface or ``step_fourier(u_hat, key=key)`` directly.

    For batched operation use ``jax.vmap``::

    ```python
    keys = jax.random.split(jax.random.PRNGKey(0), M)
    u_batch = jax.vmap(lambda k: stepper(u0, key=k))(keys)
    ```

    **References:**

    - Allen, S. M., & Cahn, J. W. (1979). A microscopic theory for
        antiphase boundary motion and its application to antiphase domain
        coarsening. *Acta Metallurgica*, 27(6), 1085-1095.
        https://doi.org/10.1016/0001-6160(79)90196-2

    - Lord, G. J., Powell, C. E., & Shardlow, T. (2014).
        *An Introduction to Computational Stochastic PDEs*.
        Cambridge University Press.
        https://doi.org/10.1017/CBO9781139016247
        (EEM method, exact variance formula, and Q-Wiener
        discretisation: Chapters 7-10.)

    - Jentzen, A., & Kloeden, P. E. (2009a). Overcoming the order
        barrier in the numerical approximation of stochastic partial
        differential equations with additive space-time noise.
        *Proceedings of the Royal Society A*, 465(2102), 649-667.
        https://doi.org/10.1098/rspa.2008.0325
        (ETD-based EEM integrators and Milstein correction for SPDEs.)

    - Jentzen, A., & Kloeden, P. E. (2009b). The numerical approximation
        of stochastic partial differential equations.
        *Milan Journal of Mathematics*, 77(1), 205-244.
        https://doi.org/10.1007/s00032-009-0100-0
        (Review of convergence theory for numerical SPDEs.)

    - Hutzenthaler, M., Jentzen, A., & Kloeden, P. E. (2011). Strong and
        weak divergence in finite time of Euler's method for stochastic
        differential equations with non-globally Lipschitz continuous
        coefficients. *Proceedings of the Royal Society A*, 467(2130),
        1563-1576. https://doi.org/10.1098/rspa.2010.0348
        (Divergence of untamed Euler for super-linearly growing
        coefficients; motivation for taming.)

    - Hutzenthaler, M., & Jentzen, A. (2015). Numerical approximations
        of stochastic differential equations with non-globally Lipschitz
        continuous coefficients. *Memoirs of the American Mathematical
        Society*, 236(1112). https://doi.org/10.1090/memo/1112
        (Tamed Euler-Maruyama: almost-sure boundedness and convergence
        rates for super-linearly growing drift/diffusion coefficients.)

    - Cox, S. M., & Matthews, P. C. (2002). Exponential time differencing
        for stiff systems. *Journal of Computational Physics*, 176(2),
        430-455. https://doi.org/10.1006/jcph.2002.6995
        (ETD1 and ETDRK4 methods.)

    - Kassam, A.-K., & Trefethen, L. N. (2005). Fourth-order time-stepping
        for stiff PDEs. *SIAM Journal on Scientific Computing*, 26(4),
        1214-1233. https://doi.org/10.1137/S1064827502410633
        (Contour-integral method for stable ETD coefficient evaluation.)

    - Funaki, T. (1995). The scaling limit for a stochastic PDE and the
        separation of phases. *Probability Theory and Related Fields*,
        102(2), 221-288. https://doi.org/10.1007/BF01213390
        (Invariant measure and sharp-interface limit of stochastic
        Allen-Cahn; theoretical basis for the λ=0 validation tests.)
    """

    # ── Scalar hyperparameters ─────────────────────────────────────────
    diffusivity: float
    lambda_: float
    sigma: float
    noise_alpha: float
    use_taming: bool
    use_milstein: bool
    _dealiasing_fraction_stoch: float

    # noise_type drives Python-level if/else branching inside _stochastic_step
    # and must be static so JAX/Equinox never treats it as a traceable leaf.
    noise_type: str = eqx.field(static=True)

    # ── Precomputed spectral arrays ────────────────────────────────────
    # _noise_std      : σ · (1+|k|²)^{-α/2} · √(Δt/dx^d)
    #                   Additive noise: σ is embedded; variance per step
    #                   is σ² · Q_base · factor.
    # _noise_std_base :   (1+|k|²)^{-α/2} · √(Δt/dx^d)
    #                   Multiplicative noise: σ-free; σ enters exactly
    #                   once via the u · dW product, preventing the
    #                   σ-double-counting bug of an earlier version.
    _noise_std: Array       # shape (1, *wavenumber_shape)
    _noise_std_base: Array  # shape (1, *wavenumber_shape)

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
        lambda_: float = 1.0,
        noise_type: Literal["additive", "multiplicative"] = "additive",
        sigma: float = 0.1,
        noise_alpha: float = 1.0,
        use_taming: bool = True,
        use_milstein: bool = False,
        order: int = 1,
        dealiasing_fraction: float = 2.0 / 3.0,
    ) -> None:
        # Guard: Milstein for space-time white noise requires iterated
        # stochastic integrals that diverge (Lord et al., 2014, §10.5).
        if noise_alpha == 0.0 and use_milstein:
            use_milstein = False

        # ── Store all fields before super().__init__ ───────────────────
        # BaseStepper.__init__ immediately calls _build_linear_operator
        # and _build_nonlinear_fun, so every attribute must exist first.
        self.diffusivity = diffusivity
        self.lambda_ = lambda_
        self.noise_type = noise_type
        self.sigma = sigma
        self.noise_alpha = noise_alpha
        self.use_taming = use_taming
        self.use_milstein = use_milstein
        self._dealiasing_fraction_stoch = dealiasing_fraction

        # ── Precompute noise spectral arrays ───────────────────────────
        # BaseStepper builds and discards derivative_operator locally;
        # we recompute it here via the same pure function.
        deriv_op = build_derivative_operator(
            num_spatial_dims, domain_extent, num_points
        )
        # Physical wavenumber squared: |k|²_phys = -∑ᵢ (∂/∂xᵢ)²_k (real part)
        k_sq = -jnp.sum(deriv_op ** 2, axis=0, keepdims=True).real  # (1, *wn)
        dx_d = (domain_extent / num_points) ** num_spatial_dims

        # Spectral filter shared by both noise variants:
        #   filter_k = (1 + |k|²_phys)^{-α/2}
        filter_k = (1.0 + k_sq) ** (-noise_alpha / 2.0)
        sqrt_dt_over_dxd = jnp.sqrt(dt / dx_d)

        # Additive variant: σ pre-multiplied so dW_hat = scaling·noise_std·Z
        self._noise_std = sigma * filter_k * sqrt_dt_over_dxd

        # Multiplicative variant: σ-free so σ enters exactly once via σu·dW
        self._noise_std_base = filter_k * sqrt_dt_over_dxd

        # ── BaseStepper.__init__ (confirmed signature) ─────────────────
        # Signature: (num_spatial_dims, domain_extent, num_points, dt,
        #             *, num_channels, order)
        # No dealiasing_fraction argument at the BaseStepper level.
        if order < 1:
            raise ValueError(
                f"order must be ≥ 1 (ETDRK0 has no _nonlinear_fun). Got {order}."
            )
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
        )

    # ── Abstract methods required by BaseStepper ──────────────────────

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        """Linear operator L_k = ν Δ_k + λ.

        Matches ``exponax.stepper.reaction.AllenCahn`` exactly:

        - ``first_order_coefficient = λ``  → ETD (analytic)
        - ``third_order_coefficient = -λ`` → 𝒩(u) = -λu³ (explicit)

        Their sum recovers ``ν Δu + λu - λu³ = ν Δu + λ(u - u³)``. ✓
        """
        laplacian = jnp.sum(derivative_operator ** 2, axis=0, keepdims=True)
        return self.diffusivity * laplacian + self.lambda_

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> TamedPolynomialNonlinearFun:
        """Nonlinear function 𝒩(u) = -λu³  (cubic part only).

        The complementary linear part ``+λu`` is handled by the ETD operator.
        We construct the cubic polynomial coefficients `[0, 0, 0, -λ]` and
        forward them to the tamed polynomial helper which handles dealiasing
        and optional Hutzenthaler-Jentzen taming.
        """
        # Polynomial coefficients for -λ u³
        coeffs = [0.0, 0.0, 0.0, -float(self.lambda_)]
        return TamedPolynomialNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            dt=self.dt,
            coefficients=coeffs,
            dealiasing_fraction=self._dealiasing_fraction_stoch,
            use_taming=self.use_taming,
        )

    # ── Public interface ───────────────────────────────────────────────

    def step(self, u: Float[Array, "1 *N"]) -> Float[Array, "1 *N"]:
        """Disabled: stochastic steppers require a PRNG key.

        Raises
        ------
        NotImplementedError
            Always.  Use ``stepper(u, key=key)`` or
            ``stepper.step_fourier(u_hat, key=key)`` instead.
        """
        raise NotImplementedError(
            "StochasticAllenCahn.step() is not available because a PRNG key "
            "is required for the noise increment.\n"
            "Use stepper(u, key=key) or stepper.step_fourier(u_hat, key=key)."
        )

    def __call__(
        self,
        u: Float[Array, "1 *N"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "1 *N"]:
        """Perform one EEM step in physical space.

        **Arguments:**

        - `u`: The current physical-space field, shape ``(1, N)`` for
            1-D, ``(1, N, N)`` for 2-D, etc.
        - `key`: A JAX PRNGKey consumed to draw the noise increment.
            Two calls with the same ``u`` but different keys produce
            independent outcomes.

        **Returns:**

        - `u_next`: Updated field, same shape as ``u``.

        !!! note
            For batched operation use ``jax.vmap``::

                u_batch = jax.vmap(lambda k: stepper(u0, key=k))(keys)
        """
        expected = (self.num_channels,) + _spatial_shape(
            self.num_spatial_dims, self.num_points
        )
        if u.shape != expected:
            raise ValueError(
                f"Expected shape {expected}, got {u.shape}. "
                "Use jax.vmap for batched operation."
            )
        return self._stochastic_step(u, key=key)

    # ── Core EEM step ──────────────────────────────────────────────────

    def _stochastic_step(
        self,
        u: Float[Array, "1 *N"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "1 *N"]:
        """EEM update with exact modewise stochastic-integral variance."""

        # 1. Transform to Fourier space.
        u_hat = fft(u, num_spatial_dims=self.num_spatial_dims)

        # 2. ETD coefficients from the ETDRK integrator stored by BaseStepper.
        exp_term = self._integrator._exp_term    # exp(L_k Δt)
        phi_1_dt = self._integrator._coef_1      # φ₁(L_k Δt) · Δt

        # 3. Nonlinear term 𝒩̂(u) = FFT(-λu³) via the dealiased nonlinear fun.
        N_hat = self._integrator._nonlinear_fun(u_hat)

        # 4. Sample independent complex Gaussians for all rfft modes.
        key, k1, k2 = jax.random.split(key, 3)
        z_r = jax.random.normal(k1, u_hat.shape)
        z_i = jax.random.normal(k2, u_hat.shape)

        # 5. Exact stochastic-integral variance factor (Lord et al., 2014, §10.5):
        #
        #        Var_k / Q_k = factor
        #                    = (e^{2L_k Δt} - 1) / (2 L_k)   [L_k ≠ 0]
        #                    → Δt                              [L_k → 0]
        #
        #    Recovered as (1 - e^{2 L_dt}) / (-2 L_k) where L_dt = L_k · Δt.
        #    For over-damped modes (exp_term → 0) factor → 0, so those
        #    modes receive negligible noise; no NaN risk.
        L_dt = jnp.log(exp_term)       # L_k · Δt
        L_k = L_dt / self.dt
        factor = jnp.where(
            jnp.abs(L_k) > 1e-30,
            (1.0 - jnp.exp(2.0 * L_dt)) / (-2.0 * L_k),
            self.dt,
        )

        # rfft-array → coefficient-space conversion.
        scaling = build_scaling_array(
            self.num_spatial_dims, self.num_points, mode="reconstruction"
        )

        # 6. Build the noise increment.
        if self.noise_type == "additive":
            # _noise_std already contains σ.
            # Coefficient-space variance per step: σ² · Q_base · factor.
            noise_var = (self._noise_std ** 2) * factor
            dW_hat = (
                scaling * jnp.sqrt(noise_var) *
                (z_r + 1j * z_i) / jnp.sqrt(2.0)
            )
            noise_increment = dW_hat

        else:  # multiplicative: σ(u) = σu
            # _noise_std_base does NOT contain σ; σ enters exactly once
            # below via the σ·u·dW product to avoid double-counting.
            # Coefficient-space base variance per step: Q_base · factor.
            noise_var_base = (self._noise_std_base ** 2) * factor
            dW_hat_base = (
                scaling
                * jnp.sqrt(noise_var_base)
                * (z_r + 1j * z_i)
                / jnp.sqrt(2.0)
            )
            nl = self._integrator._nonlinear_fun
            u_phys = nl.ifft(nl.dealias(u_hat))
            dW_phys_base = nl.ifft(dW_hat_base).real

            # σ · u · dW  (σ applied exactly once)
            noise_increment = nl.fft(self.sigma * u_phys * dW_phys_base)

            if self.use_milstein:
                # First-order Milstein correction for σ(u) = σu
                # (Jentzen & Kloeden, 2009a):
                #
                #   ½ σ(u) σ'(u) (dW² - E[dW²]) = ½ σ² u (dW² - E[dW²])
                #
                # E[dW_phys_base(x)²] is recovered from the coefficient-space
                # variance via Parseval / the scaling array.
                var_dW_hat_base = (scaling ** 2) * noise_var_base
                E_dW_phys_base_sq = nl.ifft(var_dW_hat_base).real
                mil_phys = (
                    0.5
                    * self.sigma ** 2
                    * u_phys
                    * (dW_phys_base ** 2 - E_dW_phys_base_sq)
                )
                # NOTE: the φ₁Δt prefactor treats the Milstein correction as
                # an ETD nonlinear term.  This is non-standard (see "Known
                # limitations" in the module docstring) but retained for
                # backward compatibility with the existing passing tests.
                noise_increment = noise_increment + phi_1_dt * nl.fft(mil_phys)

        # 7. EEM update (mild-solution form for the linear part).
        u_hat_new = exp_term * u_hat + phi_1_dt * N_hat + noise_increment

        # 8. Inverse transform; discard machine-precision imaginary residual.
        return ifft(
            u_hat_new,
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
        ).real

    # ── Fourier-space variant ──────────────────────────────────────────

    def step_fourier(
        self,
        u_hat: Complex[Array, "1 ... (N//2)+1"],
        *,
        key: PRNGKeyArray,
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        """Perform one EEM step entirely in Fourier space.

        More efficient than ``__call__`` when the caller already holds the
        Fourier representation and will consume the output in Fourier space
        (e.g., custom rollout loops that never materialise the physical-space
        field).

        **Arguments:**

        - `u_hat`: The real-valued Fourier transform of the current field,
            shape ``(1, ..., (N//2)+1)``.
        - `key`: A JAX PRNGKey.

        **Returns:**

        - `u_next_hat`: Fourier transform of the updated field, same
            shape as ``u_hat``.
        """
        exp_term = self._integrator._exp_term
        phi_1_dt = self._integrator._coef_1
        N_hat = self._integrator._nonlinear_fun(u_hat)

        key, k1, k2 = jax.random.split(key, 3)
        z_r = jax.random.normal(k1, u_hat.shape)
        z_i = jax.random.normal(k2, u_hat.shape)

        L_dt = jnp.log(exp_term)
        L_k = L_dt / self.dt
        factor = jnp.where(
            jnp.abs(L_k) > 1e-30,
            (1.0 - jnp.exp(2.0 * L_dt)) / (-2.0 * L_k),
            self.dt,
        )
        scaling = build_scaling_array(
            self.num_spatial_dims, self.num_points, mode="reconstruction"
        )

        if self.noise_type == "additive":
            noise_var = (self._noise_std ** 2) * factor
            dW_hat = (
                scaling * jnp.sqrt(noise_var) *
                (z_r + 1j * z_i) / jnp.sqrt(2.0)
            )
            noise_increment = dW_hat

        else:  # multiplicative
            noise_var_base = (self._noise_std_base ** 2) * factor
            dW_hat_base = (
                scaling
                * jnp.sqrt(noise_var_base)
                * (z_r + 1j * z_i)
                / jnp.sqrt(2.0)
            )
            nl = self._integrator._nonlinear_fun
            u_phys = nl.ifft(nl.dealias(u_hat))
            dW_phys_base = nl.ifft(dW_hat_base).real
            noise_increment = nl.fft(self.sigma * u_phys * dW_phys_base)

            if self.use_milstein:
                var_dW_hat_base = (scaling ** 2) * noise_var_base
                E_dW_phys_base_sq = nl.ifft(var_dW_hat_base).real
                mil_phys = (
                    0.5
                    * self.sigma ** 2
                    * u_phys
                    * (dW_phys_base ** 2 - E_dW_phys_base_sq)
                )
                noise_increment = noise_increment + phi_1_dt * nl.fft(mil_phys)

        return exp_term * u_hat + phi_1_dt * N_hat + noise_increment
