from typing import TypeVar, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float

from .._utils import make_grid, wrap_bc

N = TypeVar("N")


def plot_state_1d(
    state: Float[Array, "C N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    labels: list[str] = None,
    ax=None,
    xlabel: str = "Space",
    ylabel: str = "Value",
    **kwargs,
):
    """
    Plot the state of a 1d field.

    Requires the input to be a real array with two axis: a leading channel axis
    and a spatial axis.

    **Arguments:**

    - `state`: The state to plot as a two axis array. If there is more than one
        dimension in the first axis (i.e., multiple channels) then each channel
        will be plotted in a different color. Use the `labels` argument to
        provide a legend.
    - `vlim`: The limits of the y-axis.
    - `domain_extent`: The extent of the spatial domain. If not provided, the
        domain extent will be the number of points in the spatial axis. This
        adjusts the x-axis.
    - `labels`: The labels for the legend. This should be a list of strings with
        the same length as the number of channels.
    - `ax`: The axis to plot on. If not provided, a new figure will be created.
    - `**kwargs`: Additional arguments to pass to the plot function.

    **Returns:**

    - If `ax` is not provided, returns a tuple with the figure, axis, and plot
        object. Otherwise, returns the plot object.
    """
    if state.ndim != 2:
        raise ValueError("state must be a two-axis array.")

    state_wrapped = wrap_bc(state)

    num_points = state.shape[-1]

    if domain_extent is None:
        # One more because we wrapped the BC
        domain_extent = num_points

    grid = make_grid(1, domain_extent, num_points, full=True)

    if ax is None:
        fig, ax = plt.subplots()

    p = ax.plot(grid[0], state_wrapped.T, label=labels, **kwargs)
    ax.set_ylim(vlim)
    ax.grid()
    if labels is not None:
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ax is None:
        return fig, ax, p
    else:
        return p


def plot_state_1d_facet(
    states: Float[Array, "B C N"],
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
    Plot a facet of 1d states.

    Requires the input to be a real array with three axis: a leading batch axis,
    a channel axis, and a spatial axis. Dimensions in the batch axis will be
    distributed over the individual plots. Dimensions in the channel axis will
    be plotted in different colors.

    **Arguments:**

    - `states`: The states to plot as a three axis array. If there is more than
        one dimension in the channel axis (i.e., multiple channels) then each
        channel will be plotted in a different color. Use the `labels` argument
        to provide a legend. Use the `titles` argument to provide titles for each
        plot.
    - `vlim`: The limits of the y-axis.
    - `labels`: The labels for the legend. This should be a list of strings with
        the same length as the number of channels.
    - `titles`: The titles for each plot. This should be a list of strings with
        the same length as the number of states.
    - `domain_extent`: The extent of the spatial domain. If not provided, the
        domain extent will be the number of points in the spatial axis. This
        adjusts the x-axis.
    - `grid`: The grid layout for the facet plot. This should be a tuple with
        two integers. If the number of states is less than the product of the
        grid, the remaining axes will be removed.
    - `figsize`: The size of the figure.
    - `**kwargs`: Additional arguments to pass to the plot
        function.

    **Returns:**

    - The figure.
    """
    if states.ndim != 3:
        raise ValueError("states must be a three-axis array.")

    fig, ax_s = plt.subplots(*grid, figsize=figsize)

    num_batches = states.shape[0]

    for i, ax in enumerate(ax_s.flatten()):
        if i < num_batches:
            plot_state_1d(
                states[i],
                vlim=vlim,
                domain_extent=domain_extent,
                labels=labels,
                ax=ax,
                **kwargs,
            )
            if titles is not None:
                ax.set_title(titles[i])
        else:
            ax.remove()

    return fig


def plot_spatio_temporal(
    trj: Float[Array, "T 1 N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    ax=None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    if trj.ndim != 3:
        raise ValueError("trj must be a two-axis array.")

    trj_wrapped = jax.vmap(wrap_bc)(trj)

    if domain_extent is not None:
        space_range = (0, domain_extent)
    else:
        # One more because we wrapped the BC
        space_range = (0, trj_wrapped.shape[1])

    if dt is not None:
        time_range = (0, dt * trj_wrapped.shape[0])
        if not include_init:
            time_range = (dt, time_range[1])
    else:
        time_range = (0, trj_wrapped.shape[0] - 1)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(
        trj_wrapped[:, 0, :].T,
        vmin=vlim[0],
        vmax=vlim[1],
        cmap="RdBu_r",
        origin="lower",
        aspect="auto",
        extent=(*time_range, *space_range),
        **kwargs,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Space")

    return im


def plot_spatio_temporal_facet(
    trjs: Union[Float[Array, "T C N"], Float[Array, "B T 1 N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles: list[str] = None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    if facet_over_channels:
        if trjs.ndim != 3:
            raise ValueError("trjs must be a three-axis array.")
    else:
        if trjs.ndim != 4:
            raise ValueError("trjs must be a four-axis array.")

    fig, ax_s = plt.subplots(*grid, sharex=True, sharey=True, figsize=figsize)

    if facet_over_channels:
        trjs = jnp.swapaxes(trjs, 0, 1)
        trjs = trjs[:, :, None, :]

    num_subplots = trjs.shape[0]

    for i, ax in enumerate(ax_s.flatten()):
        single_trj = trjs[i]
        plot_spatio_temporal(
            single_trj,
            vlim=vlim,
            ax=ax,
            domain_extent=domain_extent,
            dt=dt,
            include_init=include_init,
            **kwargs,
        )
        if i >= num_subplots:
            ax.remove()
        else:
            if titles is not None:
                ax.set_title(titles[i])

    return fig


def plot_state_2d(
    state: Float[Array, "1 N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    ax=None,
    **kwargs,
):
    if domain_extent is not None:
        space_range = (0, domain_extent)
    else:
        # One more because we wrapped the BC
        space_range = (0, state.shape[-1])

    state_wrapped = wrap_bc(state)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(
        state_wrapped.T,
        vmin=vlim[0],
        vmax=vlim[1],
        cmap="RdBu_r",
        origin="lower",
        aspect="auto",
        extent=(*space_range, *space_range),
        **kwargs,
    )
    ax.set_xlabel("x_0")
    ax.set_ylabel("x_1")
    ax.set_aspect("equal")

    return im


def plot_state_2d_facet(
    states: Union[Float[Array, "B N N"], Float[Array, "B 1 N N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles: list[str] = None,
    domain_extent: float = None,
    **kwargs,
):
    if facet_over_channels:
        if states.ndim != 3:
            raise ValueError("states must be a three-axis array.")
    else:
        if states.ndim != 4:
            raise ValueError("states must be a four-axis array.")

    fig, ax_s = plt.subplots(*grid, sharex=True, sharey=True, figsize=figsize)

    for i, ax in enumerate(ax_s.flatten()):
        plot_state_2d(
            states[i],
            vlim=vlim,
            ax=ax,
            domain_extent=domain_extent,
            **kwargs,
        )
        if titles is not None:
            ax.set_title(titles[i])

    return fig
