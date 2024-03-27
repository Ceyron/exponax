"""
A collection of simple visualization tools, build on top of Matplotlib;
including animations.

You do not have to use them, as all states are pure jax arrays, plotting with
any library is straightforward.

Supported visualization methods:
    - Display 1d states as line plots
    - Display 1d trajectories as spatio-temporal image plots
    - Display 2d states as image plots

All the methods also have a `facet` version, which allows you to plot multiple
states at once.

All plotting routines (three main routines and their three facet counterparts)
can be animated over another axis (some notion of time).
"""

from ._animate import animate_state_1d, animate_state_2d, animate_state_2d_facet
from ._plot import (
    plot_spatio_temporal,
    plot_spatio_temporal_facet,
    plot_state_1d,
    plot_state_1d_facet,
    plot_state_2d,
    plot_state_2d_facet,
)

__all__ = [
    "plot_state_1d",
    "plot_state_1d_facet",
    "plot_state_2d",
    "plot_spatio_temporal",
    "plot_state_2d_facet",
    "plot_spatio_temporal_facet",
    "animate_state_2d",
    "animate_state_1d",
    "animate_state_2d_facet",
]