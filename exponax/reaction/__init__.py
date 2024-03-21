"""
Reaction-Diffusion Steppers.

They are in their own submodule because they often differ greatly from the ones
in the `exponax.stepper` module. Oftentimes the also come with their own
nonlinear function. If not, they most often use
`expnonax.nonlin_fun.PolynomialNonlinearFun`.

They often also require carefully tuned initial conditions, whereas the other
steppers often operate on a wide range of initial conditions.
"""

from ._allen_cahn import AllenCahn

# from ._belousov_zhabotinsky import BelousovZhabotinsky
from ._cahn_hilliard import CahnHilliard
from ._fisher_kpp import FisherKPP
from ._gray_scott import GrayScott
from ._swift_hohenberg import SwiftHohenberg

__all__ = [
    "AllenCahn",
    # "BelousovZhabotinsky",
    "CahnHilliard",
    "FisherKPP",
    "GrayScott",
    "SwiftHohenberg",
]
