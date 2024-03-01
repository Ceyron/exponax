from ._base import BaseNonlinearFun
from ._convection import ConvectionNonlinearFun
from ._gradient_norm import GradientNormNonlinearFun
from ._polynomial import PolynomialNonlinearFun
from ._reaction import (
    BelousovZhabotinskyNonlinearFun,
    CahnHilliardNonlinearFun,
    GrayScottNonlinearFun,
)
from ._vorticity_convection import (
    VorticityConvection2d,
    VorticityConvection2dKolmogorov,
)
from ._zero import ZeroNonlinearFun

__all__ = [
    "BaseNonlinearFun",
    "ConvectionNonlinearFun",
    "GradientNormNonlinearFun",
    "PolynomialNonlinearFun",
    "BelousovZhabotinskyNonlinearFun",
    "CahnHilliardNonlinearFun",
    "GrayScottNonlinearFun",
    "VorticityConvection2d",
    "VorticityConvection2dKolmogorov",
    "ZeroNonlinearFun",
]
