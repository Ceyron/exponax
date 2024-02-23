from .burgers import Burgers
from .convection import GeneralConvectionStepper
from .gradient_norm import GeneralGradientNormStepper
from .korteveg_de_vries import KortevegDeVries
from .kuramoto_sivashinsky import (
    KuramotoSivashinsky,
    KuramotoSivashinskyConservative,
)
from .linear import (
    Advection,
    Diffusion,
    AdvectionDiffusion,
    Dispersion,
    HyperDiffusion,
    GeneralLinearStepper,
)
from .navier_stokes import (
    NavierStokesVorticity2d,
    KolmogorovFlowVorticity2d,
)
from .nikolaevskiy import (
    Nikolaevskiy,
    NikolaevskiyConservative,
)
from .reaction import (
    SwiftHohenberg,
    GrayScott,
    FisherKPP,
    AllenCahn,
    CahnHilliard,
    BelousovZhabotinsky,
)
