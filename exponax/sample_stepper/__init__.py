from .burgers import Burgers
from .convection import GeneralConvectionStepper
from .gradient_norm import GeneralGradientNormStepper
from .korteveg_de_vries import KortewegDeVries
from .kuramoto_sivashinsky import KuramotoSivashinsky, KuramotoSivashinskyConservative
from .linear import (
    Advection,
    AdvectionDiffusion,
    Diffusion,
    Dispersion,
    GeneralLinearStepper,
    HyperDiffusion,
)
from .navier_stokes import KolmogorovFlowVorticity2d, NavierStokesVorticity2d
from .nikolaevskiy import Nikolaevskiy, NikolaevskiyConservative
from .reaction import (
    AllenCahn,
    BelousovZhabotinsky,
    CahnHilliard,
    FisherKPP,
    GrayScott,
    SwiftHohenberg,
)

__all__ = [
    "Advection",
    "BelousovZhabotinsky",
    "Diffusion",
    "AdvectionDiffusion",
    "Dispersion",
    "HyperDiffusion",
    "Burgers",
    "KuramotoSivashinsky",
    "KuramotoSivashinskyConservative",
    "SwiftHohenberg",
    "GrayScott",
    "KortewegDeVries",
    "FisherKPP",
    "AllenCahn",
    "CahnHilliard",
    "GeneralLinearStepper",
    "GeneralConvectionStepper",
    "GeneralGradientNormStepper",
    "Nikolaevskiy",
    "NikolaevskiyConservative",
    "NavierStokesVorticity2d",
    "KolmogorovFlowVorticity2d",
]
