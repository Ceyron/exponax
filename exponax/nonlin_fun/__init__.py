from .base import BaseNonlinearFun
from .convection import ConvectionNonlinearFun
from .gradient_norm import GradientNormNonlinearFun
from .polynomial import PolynomialNonlinearFun
from .reaction import (
    BelousovZhabotinskyNonlinearFun,
    CahnHilliardNonlinearFun,
    GrayScottNonlinearFun,
)
from .vorticity_convection import VorticityConvection2d, VorticityConvection2dKolmogorov
from .zero import ZeroNonlinearFun

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
