"""
A collection of simple visualization tools, build on top of Matplotlib;
including animations.

You do not have to use them, as all states are pure jax arrays, plotting with
any library is straightforward.
"""

from ._animate import animate_state_1d, animate_state_2d, animate_state_2d_facet
from ._plot import (
    plot_spatio_temporal,
    plot_spatio_temporal_facet,
    plot_state_1d,
    plot_state_2d,
    plot_state_2d_facet,
)

__all__ = [
    "plot_state_1d",
    "plot_state_2d",
    "plot_spatio_temporal",
    "plot_state_2d_facet",
    "plot_spatio_temporal_facet",
    "animate_state_2d",
    "animate_state_1d",
    "animate_state_2d_facet",
]
