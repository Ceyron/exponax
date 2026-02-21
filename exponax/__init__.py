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

from ._stochastic_utils import (
    stochastic_rollout,
    stochastic_ensemble_rollout,
    structure_factor,
    richardson_weak_extrapolation,
    strang_split_step,
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
    "richardson_weak_extrapolation",
    "stack_sub_trajectories",
    "stochastic_ensemble_rollout",
    "stochastic_rollout",
    "strang_split_step",
    "structure_factor",
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
