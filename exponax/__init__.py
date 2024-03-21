from . import _metrics as metrics
from . import _poisson as poisson
from . import etdrk, ic, nonlin_fun, normalized, reaction, stepper
from ._base_stepper import BaseStepper
from ._forced_stepper import ForcedStepper
from ._repeated_stepper import RepeatedStepper
from ._spectral import derivative, make_incompressible
from ._utils import (
    build_ic_set,
    make_grid,
    repeat,
    rollout,
    stack_sub_trajectories,
    wrap_bc,
)
from ._viz import (  # plot_multiple_state_1d,
    make_animation,
    make_animation_1d,
    make_grouped_animation,
    plot_multiple_spatio_temporal,
    plot_spatio_temporal,
    plot_state_1d,
)

__all__ = [
    "BaseStepper",
    "ForcedStepper",
    "normalized",
    "poisson",
    "RepeatedStepper",
    "derivative",
    "make_incompressible",
    "make_grid",
    "make_animation",
    "make_animation_1d",
    "make_grouped_animation",
    "plot_state_1d",
    # "plot_multiple_state_1d",
    "plot_spatio_temporal",
    "plot_multiple_spatio_temporal",
    "rollout",
    "repeat",
    "stack_sub_trajectories",
    "stepper",
    "build_ic_set",
    "wrap_bc",
    "metrics",
    "nonlin_fun",
    "ic",
    "etdrk",
    "reaction",
]
