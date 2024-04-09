from typing import TypeVar, Union

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


def animate_state_1d_facet(
    trj: Float[Array, "T B C N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    labels: list[str] = None,
    titles: list[str] = None,
    domain_extent: float = None,
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    **kwargs,
):
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


def animate_spatio_temporal(
    trjs: Float[Array, "S T C N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
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
    if facet_over_channels:
        if trjs.ndim != 4:
            raise ValueError("trjs must be a four-axis array.")
    else:
        if trjs.ndim != 5:
            raise ValueError("states must be a five-axis array.")
    # TODO
    pass


def animate_state_2d(
    trj: Float[Array, "T 1 N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
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
    trj.shape = (n_trjs, n_timesteps, ...)
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
