"""
Collection of nonlinear functions for ETDRK methods.
Every nonlinear function is a subclass of ``BaseNonlinearFun`` and follows
the same pseudo-spectral pattern:

1. Pre-dealias in Fourier space (via ``self.ifft``, which applies the
   dealiasing mask before the inverse transform).
2. Evaluate the nonlinearity pointwise in physical space.
3. Post-dealias and return to Fourier space (via ``self.fft``, which
   applies the dealiasing mask after the forward transform).

In an n-dimensional setting the available functions are:

    1. Zero:           ``𝒩(u) = 0``
    2. Convection:     ``𝒩(u) = b₁ ½ ∇ ⋅ (u ⊗ u)``
    3. Gradient Norm:  ``𝒩(u) = b₂ ½ ‖∇u‖₂²``
    4. Polynomial:     ``𝒩(u) = ∑ᵢ cᵢ uⁱ``
    5. Vorticity Convection (2D only):
                        ``𝒩(u) = b ([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u``
    6. Projected Convection (3D only):
                        ``𝒩(u) = 𝒫(u × ω)``  with  ``ω = ∇ × u``
    7. Projected Convection with Kolmogorov forcing (3D only):
                        ``𝒩(u) = 𝒫(u × ω) + f``
    8. Leray Projection:
                        ``𝒫(u) = u - ∇(Δ⁻¹ ∇ ⋅ u)``

The zero nonlinear function is used for linear equations.

As a meta-nonlinear function, ``GeneralNonlinearFun`` combines a quadratic
polynomial, single-channel convection, and gradient norm with respective
coefficients::

```
    𝒩(u) = b₀ u² + b₁ ½ (1⃗ ⋅ ∇)(u²) + b₂ ½ ‖∇u‖₂²
```

Some reaction-diffusion equations have their own nonlinear functions found 
in their respective modules.

A nonlinear function can also encode a forcing term as done with
``VorticityConvection2dKolmogorov`` and ``ProjectedConvection3dKolmogorov``.

Tamed polynomial nonlinear function
-----------------------------------
``TamedPolynomialNonlinearFun`` evaluates a pointwise polynomial
nonlinearity

    ``𝒩(u) = ∑ᵢ cᵢ uⁱ``

in physical space using the package's pseudo-spectral convention (pre- and
post-dealiasing).

**Taming.** The cubic term grows super-linearly and can cause numerical
blowup under large noise amplitude or coarse time steps.  Setting
``use_taming=True`` (the default) activates the Hutzenthaler-Jentzen tamed
Euler-Maruyama nonlinearity::

```
    𝒩_tamed(u) = 𝒩(u) / (1 + Δt |𝒩(u)|)
```

which restores almost-sure boundedness without reducing the strong
convergence order of the integrator (Hutzenthaler, Jentzen & Kloeden,
2011; Hutzenthaler & Jentzen, 2015).  Set ``use_taming=False`` only when
an exact comparison against an untamed deterministic reference is required.

For the classical Allen-Cahn cubic nonlinearity one can pass 
``coefficients=[0,0,0,-lambda_]``.

Taming is a general stabilisation technique that is reusable by any future 
stochastic stepper whose nonlinear term grows super-linearly.
"""

from ._base import BaseNonlinearFun
from ._convection import ConvectionNonlinearFun
from ._general_nonlinear import GeneralNonlinearFun
from ._gradient_norm import GradientNormNonlinearFun
from ._leray import Leray
from ._polynomial import PolynomialNonlinearFun
from ._projected_convection import (
    ProjectedConvection3d,
    ProjectedConvection3dKolmogorov,
)
from ._tamed_polynomial import TamedPolynomialNonlinearFun
from ._vorticity_convection import (
    VorticityConvection2d,
    VorticityConvection2dKolmogorov,
)
from ._zero import ZeroNonlinearFun

__all__ = [
    "BaseNonlinearFun",
    "ConvectionNonlinearFun",
    "GeneralNonlinearFun",
    "GradientNormNonlinearFun",
    "PolynomialNonlinearFun",
    "TamedPolynomialNonlinearFun",
    "VorticityConvection2d",
    "VorticityConvection2dKolmogorov",
    "ZeroNonlinearFun",
    "Leray",
    "ProjectedConvection3d",
    "ProjectedConvection3dKolmogorov",
]
