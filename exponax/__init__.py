import importlib.metadata

from . import _poisson as poisson
from . import _spectral as spectral
from . import etdrk, ic, metrics, nonlin_fun, stepper, viz
from ._base_stepper import BaseStepper
from ._forced_stepper import ForcedStepper
from ._interpolation import FourierInterpolator, map_between_resolutions
from ._repeated_stepper import RepeatedStepper
from ._spectral import derivative, fft, get_spectrum, ifft
from ._utils import (
    build_ic_set,
    make_grid,
    repeat,
    rollout,
    stack_sub_trajectories,
    wrap_bc,
)

__version__ = importlib.metadata.version("exponax")

__all__ = [
    "BaseStepper",
    "ForcedStepper",
    "poisson",
    "RepeatedStepper",
    "derivative",
    "fft",
    "ifft",
    "get_spectrum",
    "make_grid",
    "rollout",
    "repeat",
    "stack_sub_trajectories",
    "build_ic_set",
    "wrap_bc",
    "metrics",
    "etdrk",
    "ic",
    "nonlin_fun",
    "stepper",
    "viz",
    "spectral",
    "FourierInterpolator",
    "map_between_resolutions",
]
