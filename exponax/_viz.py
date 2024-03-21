"""
Utilities for visualization.
"""

from typing import TypeVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from matplotlib.animation import FuncAnimation

from ._utils import make_grid, wrap_bc

N = TypeVar("N")


def plot_state_1d(
    state: Float[Array, "B N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    labels: list[str] = None,
    ax=None,
    **kwargs,
):
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

    return p


# def plot_multiple_state_1d(
#     states: Float[Array, "B N"],
#     *,
#     vlim: tuple[float, float] = (-1.0, 1.0),
#     labels: list[str] = None,
#     domain_extent: float = None,
# ):
#     if states.ndim != 2:
#         raise ValueError(
#             "states must be a two-axis array. Extract the channel you want to plot."
#         )

#     fig, ax = plt.subplots()
#     line_list
#     for i, state in enumerate(states):
#         plot_state_1d(
#             state,
#             vlim=vlim,
#             domain_extent=domain_extent,
#             ax=ax,
#             label=labels[i] if labels is not None else None,
#         )
#     if len(states) % 2 == 0:
#         ax.grid()

#     if labels is not None:
#         ax.legend()
#     return fig


def plot_spatio_temporal(
    trj: Float[Array, "T N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    ax=None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    if trj.ndim != 2:
        raise ValueError(
            "trj must be a two-axis array. Extract the channel you want to plot."
        )

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
        trj_wrapped.T,
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


def plot_multiple_spatio_temporal(
    trjs: Float[Array, "B T N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles: list[str] = None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    if trjs.ndim != 3:
        raise ValueError(
            "trjs must be a three-axis array. Extract the channel you want to plot."
        )

    fig, ax_s = plt.subplots(*grid, sharex=True, sharey=True, figsize=figsize)

    for i, ax in enumerate(ax_s.flatten()):
        plot_spatio_temporal(
            trjs[i],
            vlim=vlim,
            ax=ax,
            domain_extent=domain_extent,
            dt=dt,
            include_init=include_init,
            **kwargs,
        )
        if titles is not None:
            ax.set_title(titles[i])

    return fig


def make_animation_1d(
    trj: Float[Array, "T B N"],
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


def make_animation(trj, *, vlim=(-1, 1)):
    fig, ax = plt.subplots()
    im = ax.imshow(
        trj[0].squeeze().T, vmin=vlim[0], vmax=vlim[1], cmap="RdBu_r", origin="lower"
    )
    im.set_data(jnp.zeros_like(trj[0]).squeeze())

    def animate(i):
        im.set_data(trj[i].squeeze().T)
        fig.suptitle(f"t_i = {i:04d}")
        return im

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trj.shape[0], interval=100, blit=False)

    return ani


def make_grouped_animation(
    trj, *, vlim=(-1, 1), grid=(3, 3), figsize=(10, 10), titles=None
):
    """
    trj.shape = (n_trjs, n_timesteps, ...)
    """
    fig, ax_s = plt.subplots(*grid, sharex=True, sharey=True, figsize=figsize)
    im_s = []
    for i, ax in enumerate(ax_s.flatten()):
        im = ax.imshow(
            trj[i, 0].squeeze().T,
            vmin=vlim[0],
            vmax=vlim[1],
            cmap="RdBu_r",
            origin="lower",
        )
        im.set_data(jnp.zeros_like(trj[i, 0]).squeeze())
        im_s.append(im)

    def animate(i):
        for j, im in enumerate(im_s):
            im.set_data(trj[j, i].squeeze().T)
            if titles is not None:
                ax_s.flatten()[j].set_title(titles[j])
        fig.suptitle(f"t_i = {i:04d}")
        return im_s

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trj.shape[1], interval=100, blit=False)

    return ani
