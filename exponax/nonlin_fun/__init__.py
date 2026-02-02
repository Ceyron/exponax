"""
Collection of nonlinear functions for ETDRK mehtods.

In an n-dimensional setting they are:
    1. Zero: `𝒩(u) = 0`
    2. Convection: `𝒩(u) = b₁ 1/2 ∇ ⋅ (u ⊗ u)`
    3. Gradient Norm: `𝒩(u) = b₂ 1/2 ‖∇u‖₂²`
    4. Polynomial: `𝒩(u) = ∑ᵢ cᵢ uⁱ`
    5. Vorticity Convection (only 2d): `𝒩(u) = b ([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u`

The zero nonlinear function is used for linear equations.

As a meta nonlinear function, there is the General Nonlinear Function that
combines a quadratic polynomial, single-channel convection, and gradient norm
with respective coefficients:

```
    𝒩(u) = b₀ u² + b₁ 1/2 (1⃗ ⋅ ∇)(u²) + b₂ 1/2 ‖∇u‖₂²
```

Some reaction-diffusion equations have their own nonlinear functions that are
found in their respective modules.

A nonlinear function can also be used to encode a forcing term in the equation
as done with `VorticityConvection2dKolmogorov`.
"""
from ._base import BaseNonlinearFun
from ._convection import ConvectionNonlinearFun
from ._general_nonlinear import GeneralNonlinearFun
from ._gradient_norm import GradientNormNonlinearFun
from ._leray import Leray
from ._polynomial import PolynomialNonlinearFun
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
    "VorticityConvection2d",
    "VorticityConvection2dKolmogorov",
    "ZeroNonlinearFun",
    "Leray",
]
