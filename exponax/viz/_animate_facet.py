from typing import TypeVar, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from matplotlib.animation import FuncAnimation

from ._plot import plot_state_1d, plot_state_2d

N = TypeVar("N")


def animate_state_1d_facet(
    trj: Float[Array, "B T C N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    labels: list[str] = None,
    titles: list[str] = None,
    domain_extent: float = None,
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    **kwargs,
):
    """
    Animate a trajectory of faceted 1d states.

    Requires the input to be a four-axis array with a leading batch axis, a time
    axis, a channel axis, and a spatial axis. If there is more than one
    dimension in the channel axis, this will be plotted in a different color.
    Hence, there are two ways to display multiple states: either via the batch
    axis (resulting in faceted subplots) or via the channel axis (resulting in
    different colors).

    Periodic boundary conditions will be applied to the spatial axis (the state
    is wrapped around).

    **Arguments**:

    - `trj`: The trajectory of states to animate. Must be a four-axis array with
        shape `(n_batches, n_timesteps, n_channels, n_spatial)`. If the channel
        axis has more than one dimension, the different channels will be plotted
        in different colors.
    - `vlim`: The limits of the colorbar. Default is `(-1, 1)`.
    - `labels`: The labels for each channel. Default is `None`.
    - `titles`: The titles for each subplot. Default is `None`.
    - `domain_extent`: The extent of the spatial domain. Default is `None`. This
        affects the x-axis limits of the plot.
    - `grid`: The grid of subplots. Default is `(3, 3)`.
    - `figsize`: The size of the figure. Default is `(10, 10)`.
    - `**kwargs`: Additional keyword arguments to pass to the plotting function.

    **Returns**:

    - `ani`: The animation object.
    """
    if trj.ndim != 4:
        raise ValueError("states must be a four-axis array.")

    fig, ax_s = plt.subplots(*grid, figsize=figsize)

    num_subplots = trj.shape[0]

    for j, ax in enumerate(ax_s.flatten()):
        plot_state_1d(
            trj[j, 0],
            vlim=vlim,
            domain_extent=domain_extent,
            labels=labels,
            ax=ax,
            **kwargs,
        )
        if j >= num_subplots:
            ax.remove()
        else:
            if titles is not None:
                ax.set_title(titles[j])

    def animate(i):
        for j, ax in enumerate(ax_s.flatten()):
            ax.clear()
            plot_state_1d(
                trj[j, i],
                vlim=vlim,
                domain_extent=domain_extent,
                labels=labels,
                ax=ax,
                **kwargs,
            )
            if j >= num_subplots:
                ax.remove()
            else:
                if titles is not None:
                    ax.set_title(titles[j])

    ani = FuncAnimation(fig, animate, frames=trj.shape[1], interval=100, blit=False)

    return ani


def animate_spatial_temporal_facet(
    trjs: Union[Float[Array, "S T C N"], Float[Array, "B S T 1 N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    **kwargs,
):
    """
    Animate a facet of trajectories of spatio-temporal states. Allows to
    visualize "two time dimensions". One time dimension is the x-axis. The other
    is via the animation. For instance, this can be used to present how neural
    predictors learn spatio-temporal dynamics over time. The additional faceting
    dimension can be used two compare multiple networks with one another.

    Requires the input to be either a four-axis array or a five-axis array:

    - If `facet_over_channels` is `True`, the input must be a four-axis array
        with a leading outer time axis, a time axis, a channel axis, and a
        spatial axis. Each faceted subplot displays a different channel.
    - If `facet_over_channels` is `False`, the input must be a five-axis array
        with a leading batch axis, an outer time axis, a time axis, a channel
        axis, and a spatial axis. Each faceted subplot displays a different
        batch, only the zeroth dimension in the channel axis is plotted.

    Periodic boundary conditions will be applied to the spatial axis (the state
    is wrapped around).

    **Arguments**:

    - `trjs`: The trajectory of states to animate. Must be a four-axis array
        with shape `(n_timesteps_outer, n_time_steps, n_channels, n_spatial)` if
        `facet_over_channels` is `True`, or a five-axis array with shape
        `(n_batches, n_timesteps_outer, n_time_steps, n_channels, n_spatial)` if
        `facet_over_channels` is `False`.
    - `facet_over_channels`: Whether to facet over the channel axis or the batch
        axis. Default is `True`.
    - `vlim`: The limits of the colorbar. Default is `(-1, 1)`.
    - `domain_extent`: The extent of the spatial domain. Default is `None`. This
        affects the x-axis limits of the plot.
    - `dt`: The time step between each frame. Default is `None`. If provided,
        a title will be displayed with the current time. If not provided, just
        the frames are counted.
    - `include_init`: Whether to the state starts at an initial condition (t=0)
        or at the first frame in the trajectory. This affects is the the time
        range is [0, (T-1)dt] or [dt, Tdt]. Default is `False`.
    - `grid`: The grid of subplots. Default is `(3, 3)`.
    - `figsize`: The size of the figure. Default is `(10, 10)`.
    - `**kwargs`: Additional keyword arguments to pass to the plotting function.

    **Returns**:

    - `ani`: The animation object.
    """
    if facet_over_channels:
        if trjs.ndim != 4:
            raise ValueError("trjs must be a four-axis array.")
    else:
        if trjs.ndim != 5:
            raise ValueError("states must be a five-axis array.")
    # TODO
    raise NotImplementedError("Not implemented yet.")


def animate_state_2d_facet(
    trj: Union[Float[Array, "T C N N"], Float[Array, "B T 1 N N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles=None,
):
    """
    Animate a facet of trajectories of 2d states.

    Requires the input to be either a four-axis array or a five-axis array:

    - If `facet_over_channels` is `True`, the input must be a four-axis array
        with a leading time axis, a channel axis, and two spatial axes. Each
        faceted subplot displays a different channel.
    - If `facet_over_channels` is `False`, the input must be a five-axis array
        with a leading batch axis, a time axis, a channel axis, and two spatial
        axes. Each faceted subplot displays a different batch. Only the zeroth
        dimension in the channel axis is plotted.

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments**:

    - `trj`: The trajectory of states to animate. Must be a four-axis array with
        shape `(n_timesteps, n_channels, n_spatial, n_spatial)` if
        `facet_over_channels` is `True`, or a five-axis array with shape
        `(n_batches, n_timesteps, n_channels, n_spatial, n_spatial)` if
        `facet_over_channels` is `False`.
    - `facet_over_channels`: Whether to facet over the channel axis or the batch
        axis. Default is `True`.
    - `vlim`: The limits of the colorbar. Default is `(-1, 1)`.
    - `grid`: The grid of subplots. Default is `(3, 3)`.
    - `figsize`: The size of the figure. Default is `(10, 10)`.
    - `titles`: The titles for each subplot. Default is `None`.

    **Returns**:

    - `ani`: The animation object.
    """
    if facet_over_channels:
        if trj.ndim != 4:
            raise ValueError("trj must be a four-axis array.")
    else:
        if trj.ndim != 5:
            raise ValueError("trj must be a five-axis array.")

    if facet_over_channels:
        trj = jnp.swapaxes(trj, 0, 1)
        trj = trj[:, :, None]

    fig, ax_s = plt.subplots(*grid, sharex=True, sharey=True, figsize=figsize)

    for j, ax in enumerate(ax_s.flatten()):
        plot_state_2d(
            trj[j, 0],
            vlim=vlim,
            ax=ax,
        )
        if titles is not None:
            ax.set_title(titles[j])

    def animate(i):
        for j, ax in enumerate(ax_s.flatten()):
            ax.clear()
            plot_state_2d(
                trj[j, i],
                vlim=vlim,
                ax=ax,
            )
            if titles is not None:
                ax.set_title(titles[j])

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trj.shape[1], interval=100, blit=False)

    return ani
