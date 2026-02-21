"""
Physical Time Steppers associated with certain semi-linear PDEs on periodic
domains.

There are two kinds:
    1. Steppers associated with concrete PDEs
    2. Steppers that allow for flexibly defining a wide range of dynamics

The concrete PDE steppers are:
    - Advection
    - Diffusion
    - AdvectionDiffusion
    - Dispersion
    - HyperDiffusion
    - Wave
    - Burgers
    - KortewegDeVries
    - KuramotoSivashinsky
    - KuramotoSivashinskyConservative
    - NavierStokesVorticity (2D, streamfunction-vorticity formulation)
    - KolmogorovFlowVorticity (2D, streamfunction-vorticity formulation)
    - NavierStokesVelocity (3D, velocity formulation with Leray projection)
    - KolmogorovFlowVelocity (3D, velocity formulation with Leray projection)

The flexible steppers are:
    - GeneralLinearStepper: combines an arbitrary number of (isotropic) linear
      terms
    - GeneralConvectionStepper: combines a scalable convection nonlinearity with
      an arbitrary number of (isotropic) linear terms
    - GeneralGradientNormStepper: combines a gradient norm nonlinearity with an
      arbitrary number of (isotropic) linear terms
    - GeneralVorticityConvectionStepper: combines a vorticity convection (only
      works in 2d) with an arbitrary number of (isotropic) linear terms
    - GeneralPolynomialStepper: combines an arbitrary polynomial nonlinearity
      with an arbitrary number of (isotropic) linear terms
    - GeneralNonlinearStepper: combines an arbitrary scalable combination of
      three major nonlinearities (quadratic, single-channel convection, and
      gradient norm) with an arbitrary number of (isotropic) linear terms

All steppers that include the convection nonlinearity (Burgers, KortewegDeVries,
KuramotoSivashinskyConservative, and GeneralConvectionStepper) can be put into
"single-channel" mode, a simple hack with which the number of channels do not
grow with the number of spatial dimensions.

As such, the (isotropic) versions of Advection, Diffusion, AdvectionDiffusion,
Dispersion, and HyperDiffusion are special cases of GeneralLinearStepper.

The Burgers, KortewegDeVries, and KuramotoSivashinskyConservative steppers are
special cases of the GeneralConvectionStepper.

The KuramotoSivashinsky stepper is a special case of the
GeneralGradientNormStepper.

The NavierStokesVorticity and KolmogorovFlowVorticity steppers are special cases
of the GeneralVorticityConvectionStepper.

The NavierStokesVelocity and KolmogorovFlowVelocity steppers use a projected
convection nonlinearity (rotational form with Leray projection) and have no
corresponding generic stepper.

The Wave stepper uses a handcrafted diagonalization in Fourier space specific to
the wave equation. It has no corresponding generic stepper.

In the reaction submodule you find specific steppers that are special cases of
the GeneralPolynomialStepper, e.g., the FisherKPPStepper.

All of the specific steppers (except for the Navier-Stokes and Kolmogorov Flow
variants) are special cases of the GeneralNonlinearStepper (if considered
isotropic and with the convection in single-channel).


Hence, almost every (isotropic) dynamic can be expressed with the general
steppers. The specific steppers are provided for convenience and easier
accessibility for new users. Additionally, some of them also support anisotropic
modes for the linear terms.

Stochastic steppers
-------------------
The stochastic submodule provides steppers for SPDEs driven by Q-Wiener noise.
They are based on the Exponential Euler-Maruyama (EEM) method, which combines
the analytic ETD treatment of the linear part with a discrete-time stochastic
integral whose variance is exact for each resolved Fourier mode.  See
``exponax.stepper.stochastic`` for the full submodule.

    - StochasticAllenCahn: Stochastic Allen-Cahn equation on (0, L)^d,

        ∂ₜu = ν Δu + λ(u - u³) + σ(u) ξ(x, t)

      with σ(u) = σ (additive) or σ(u) = σu (multiplicative), Q-Wiener
      noise covariance Q_k ∝ (1 + |k|²)^{-α}, and optional
      Hutzenthaler-Jentzen taming of the cubic nonlinearity.

      Special cases:
        - λ = 0: stochastic heat equation (analytically tractable invariant
          measure C_k = Q_k / (2ν|k|²)).
        - σ = 0: deterministic Allen-Cahn (numerically identical to
          ``reaction.AllenCahn`` when ``use_taming=False``).

      Supported for d ∈ {1, 2, 3}.

      Calling convention::

          u_next = stepper(u, key=jax.random.PRNGKey(0))

      The ``step()`` interface is intentionally disabled (raises
      ``NotImplementedError``) because a PRNG key is always required.
      Use ``exponax.utils.stochastic_rollout`` and
      ``exponax.utils.stochastic_ensemble_rollout`` for trajectory
      generation.
"""

from . import generic as generic
from . import reaction as reaction
from . import stochastic as stochastic
from ._advection import Advection
from ._advection_diffusion import AdvectionDiffusion
from ._burgers import Burgers
from ._diffusion import Diffusion
from ._dispersion import Dispersion
from ._hyper_diffusion import HyperDiffusion
from ._korteweg_de_vries import KortewegDeVries
from ._kuramoto_sivashinsky import KuramotoSivashinsky, KuramotoSivashinskyConservative
from ._navier_stokes import (
    KolmogorovFlowVelocity,
    KolmogorovFlowVorticity,
    NavierStokesVelocity,
    NavierStokesVorticity,
)
from ._wave import Wave

__all__ = [
    "Advection",
    "Diffusion",
    "AdvectionDiffusion",
    "Dispersion",
    "HyperDiffusion",
    "Wave",
    "Burgers",
    "KortewegDeVries",
    "KuramotoSivashinsky",
    "KuramotoSivashinskyConservative",
    "NavierStokesVorticity",
    "KolmogorovFlowVorticity",
    "stochastic",
    "reaction",
    "generic",
    "NavierStokesVelocity",
    "KolmogorovFlowVelocity",
]
