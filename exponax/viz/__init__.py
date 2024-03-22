"""
A collection of simple visualization tools, build on top of Matplotlib;
including animations.

You do not have to use them, as all states are pure jax arrays, plotting with
any library is straightforward.
"""

from ._animate import make_animation, make_animation_1d, make_grouped_animation
from ._plot import (
    plot_multiple_spatio_temporal,
    plot_multiple_states_2d,
    plot_spatio_temporal,
    plot_state_1d,
    plot_state_2d,
)

__all__ = [
    "plot_state_1d",
    "plot_state_2d",
    "plot_spatio_temporal",
    "plot_multiple_states_2d",
    "plot_multiple_spatio_temporal",
    "make_animation",
    "make_animation_1d",
    "make_grouped_animation",
]
