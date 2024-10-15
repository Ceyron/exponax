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
    - Burgers
    - KortewegDeVries
    - KuramotoSivashinsky
    - KuramotoSivashinskyConservative
    - NavierStokesVorticity
    - KolmogorovFlowVorticity

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

In the reaction submodule you find specific steppers that are special cases of
the GeneralPolynomialStepper, e.g., the FisherKPPStepper.

All of the specific steppers (except for the NavierStokesVorticity and
KolmogorovFlowVorticity) are special cases of the GeneralNonlinearStepper (if
considered isotropic and with the convection in single-channel).


Hence, almost every (isotropic) dynamic can be expressed with the general
steppers. The specific steppers are provided for convenience and easier
accessiblity for new users. Additinally, some of them also support anisotropic
modes for the linear terms.
"""

from . import generic as generic
from . import reaction as reaction
from ._advection import Advection
from ._advection_diffusion import AdvectionDiffusion
from ._burgers import Burgers
from ._diffusion import Diffusion
from ._dispersion import Dispersion
from ._hyper_diffusion import HyperDiffusion
from ._korteweg_de_vries import KortewegDeVries
from ._kuramoto_sivashinsky import KuramotoSivashinsky, KuramotoSivashinskyConservative
from ._navier_stokes import KolmogorovFlowVorticity, NavierStokesVorticity

__all__ = [
    "Advection",
    "Diffusion",
    "AdvectionDiffusion",
    "Dispersion",
    "HyperDiffusion",
    "Burgers",
    "KortewegDeVries",
    "KuramotoSivashinsky",
    "KuramotoSivashinskyConservative",
    "NavierStokesVorticity",
    "KolmogorovFlowVorticity",
    "reaction",
    "generic",
]
