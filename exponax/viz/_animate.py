from typing import TypeVar

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from matplotlib.animation import FuncAnimation

from ._plot import plot_spatio_temporal, plot_state_1d, plot_state_2d

N = TypeVar("N")


def animate_state_1d(
    trj: Float[Array, "T C N"],
    *,
    vlim: tuple[float, float] = (-1, 1),
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    """
    Animate a trajectory of 1d states.

    Requires the input to be a three-axis array with a leading time axis, a
    channel axis, and a spatial axis. If there is more than one dimension in the
    channel axis, this will be plotted in a different color.

    Periodic boundary conditions will be applied to the spatial axis (the state
    is wrapped around).

    **Arguments**:

    - `trj`: The trajectory of states to animate. Must be a three-axis array
        with shape `(n_timesteps, n_channels, n_spatial)`. If the channel axis
        has more than one dimension, the different channels will be plotted in
        different colors.
    - `vlim`: The limits of the colorbar. Default is `(-1, 1)`.
    - `domain_extent`: The extent of the spatial domain. Default is `None`. This
        affects the x-axis limits of the plot.
    - `dt`: The time step between each frame. Default is `None`. If provided,
        a title will be displayed with the current time. If not provided, just
        the frames are counted.
    - `include_init`: Whether to the state starts at an initial condition (t=0)
        or at the first frame in the trajectory. This affects is the the time
        range is [0, (T-1)dt] or [dt, Tdt]. Default is `False`.
    - `**kwargs`: Additional keyword arguments to pass to the plotting function.

    **Returns**:

    - `ani`: The animation object.
    """
    fig, ax = plt.subplots()

    plot_state_1d(
        trj[0],
        vlim=vlim,
        domain_extent=domain_extent,
        ax=ax,
        **kwargs,
    )

    if include_init:
        temporal_grid = jnp.arange(trj.shape[0])
    else:
        temporal_grid = jnp.arange(1, trj.shape[0] + 1)

    if dt is not None:
        temporal_grid *= dt

    ax.set_title(f"t = {temporal_grid[0]:.2f}")

    def animate(i):
        ax.clear()
        plot_state_1d(
            trj[i],
            vlim=vlim,
            domain_extent=domain_extent,
            ax=ax,
            **kwargs,
        )

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trj.shape[0], interval=100, blit=False)

    return ani


def animate_spatio_temporal(
    trjs: Float[Array, "S T C N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    """
    Animate a trajectory of spatio-temporal states. Allows to visualize "two
    time dimensions". One time dimension is the x-axis. The other is via the
    animation. For instance, this can be used to present how neural predictors
    learn spatio-temporal dynamics over time.

    Requires the input to be a four-axis array with a leading spatial axis, a
    time axis, a channel axis, and a batch axis. Only the zeroth dimension in
    the channel axis is plotted.

    Periodic boundary conditions will be applied to the spatial axis (the state
    is wrapped around).

    **Arguments**:

    - `trjs`: The trajectory of states to animate. Must be a four-axis array
        with shape `(n_timesteps_outer, n_time_steps, n_channels, n_spatial)`.
    - `vlim`: The limits of the colorbar. Default is `(-1, 1)`.
    - `domain_extent`: The extent of the spatial domain. Default is `None`. This
        affects the x-axis limits of the plot.
    - `dt`: The time step between each frame. Default is `None`. If provided,
        a title will be displayed with the current time. If not provided, just
        the frames are counted.
    - `include_init`: Whether to the state starts at an initial condition (t=0)
        or at the first frame in the trajectory. This affects is the the time
        range is [0, (T-1)dt] or [dt, Tdt]. Default is `False`.
    - `**kwargs`: Additional keyword arguments to pass to the plotting function.

    **Returns**:

    - `ani`: The animation object.
    """
    if trjs.ndim != 4:
        raise ValueError("trjs must be a four-axis array.")

    fig, ax = plt.subplots()

    plot_spatio_temporal(
        trjs[0],
        vlim=vlim,
        domain_extent=domain_extent,
        dt=dt,
        include_init=include_init,
        ax=ax,
        **kwargs,
    )

    def animate(i):
        ax.clear()
        plot_spatio_temporal(
            trjs[i],
            vlim=vlim,
            domain_extent=domain_extent,
            dt=dt,
            include_init=include_init,
            ax=ax,
            **kwargs,
        )

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trjs.shape[0], interval=100, blit=False)

    return ani


def animate_state_2d(
    trj: Float[Array, "T 1 N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    """
    Animate a trajectory of 2d states.

    Requires the input to be a four-axis array with a leading time axis, a
    channel axis, and two spatial axes. Only the zeroth dimension in the channel
    axis is plotted.

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments**:

    - `trj`: The trajectory of states to animate. Must be a four-axis array with
        shape `(n_timesteps, 1, n_spatial, n_spatial)`.
    - `vlim`: The limits of the colorbar. Default is `(-1, 1)`.
    - `domain_extent`: The extent of the spatial domain. Default is `None`. This
        affects the x- and y-axis limits of the plot.
    - `dt`: The time step between each frame. Default is `None`. If provided,
        a title will be displayed with the current time. If not provided, just
        the frames are counted.
    - `include_init`: Whether to the state starts at an initial condition (t=0)
        or at the first frame in the trajectory. This affects is the the time
        range is [0, (T-1)dt] or [dt, Tdt]. Default is `False`.
    - `**kwargs`: Additional keyword arguments to pass to the plotting function.

    **Returns**:

    - `ani`: The animation object.
    """
    if trj.ndim != 4:
        raise ValueError("trj must be a four-axis array.")

    fig, ax = plt.subplots()

    if dt is not None:
        time_range = (0, dt * trj.shape[0])
        if not include_init:
            time_range = (dt, time_range[1])
    else:
        time_range = (0, trj.shape[0] - 1)

    plot_state_2d(
        trj[0],
        vlim=vlim,
        domain_extent=domain_extent,
        ax=ax,
    )

    def animate(i):
        ax.clear()
        plot_state_2d(
            trj[i],
            vlim=vlim,
            domain_extent=domain_extent,
            ax=ax,
        )

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trj.shape[0], interval=100, blit=False)

    return ani
