from ._burgers import Burgers
from ._convection import GeneralConvectionStepper
from ._gradient_norm import GeneralGradientNormStepper
from ._korteveg_de_vries import KortewegDeVries
from ._kuramoto_sivashinsky import KuramotoSivashinsky, KuramotoSivashinskyConservative
from ._linear import (
    Advection,
    AdvectionDiffusion,
    Diffusion,
    Dispersion,
    GeneralLinearStepper,
    HyperDiffusion,
)
from ._navier_stokes import KolmogorovFlowVorticity2d, NavierStokesVorticity2d
from ._nikolaevskiy import Nikolaevskiy, NikolaevskiyConservative
from ._reaction import (
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
