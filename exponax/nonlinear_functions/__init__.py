from .base import BaseNonlinearFun
from .convection import ConvectionNonlinearFun
from .gradient_norm import GradientNormNonlinearFun
from .polynomial import PolynomialNonlinearFun
from .reaction import (
    GrayScottNonlinearFun,
    CahnHilliardNonlinearFun,
    BelousovZhabotinskyNonlinearFun,
)
from .vorticity_convection import (
    VorticityConvection2d,
    VorticityConvection2dKolmogorov,
)
from .zero import ZeroNonlinearFun
