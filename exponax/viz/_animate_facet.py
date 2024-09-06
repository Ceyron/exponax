from typing import Literal, TypeVar, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.animation import FuncAnimation

from .._utils import wrap_bc
from ._plot import plot_state_1d, plot_state_2d
from ._volume import volume_render_state_3d, zigzag_alpha

N = TypeVar("N")


def animate_state_1d_facet(
    trj: Float[Array, "B T C N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    labels: list[str] = None,
    titles: list[str] = None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
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

    if include_init:
        temporal_grid = jnp.arange(trj.shape[1])
    else:
        temporal_grid = jnp.arange(1, trj.shape[1] + 1)

    if dt is not None:
        temporal_grid *= dt

    fig, ax_s = plt.subplots(*grid, figsize=figsize)

    num_subplots = trj.shape[0]

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
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
    title = fig.suptitle(f"t = {temporal_grid[0]:.2f}")

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
        title.set_text(f"t = {temporal_grid[i]:.2f}")

    ani = FuncAnimation(fig, animate, frames=trj.shape[1], interval=100, blit=False)

    return ani


def animate_spatio_temporal_facet(
    trjs: Union[Float[Array, "S T C N"], Float[Array, "B S T 1 N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    cmap: str = "RdBu_r",
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
    - `cmap`: The colormap to use. Default is `"RdBu_r"`.
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
    cmap: str = "RdBu_r",
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
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
    - `cmap`: The colormap to use. Default is `"RdBu_r"`.
    - `domain_extent`: The extent of the spatial domain. Default is `None`. This
        affects the x-axis and y-axis limits of the plot.
    - `dt`: The time step between each frame. Default is `None`.
    - `include_init`: Whether to the state starts at an initial condition (t=0)
        or at the first frame in the trajectory. This affects is the the time
        range is [0, (T-1)dt] or [dt, Tdt]. Default is `False`.
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

    if include_init:
        temporal_grid = jnp.arange(trj.shape[1])
    else:
        temporal_grid = jnp.arange(1, trj.shape[1] + 1)

    if dt is not None:
        temporal_grid *= dt

    fig, ax_s = plt.subplots(*grid, sharex=True, sharey=True, figsize=figsize)

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
    for j, ax in enumerate(ax_s.flatten()):
        plot_state_2d(
            trj[j, 0],
            vlim=vlim,
            cmap=cmap,
            ax=ax,
            domain_extent=domain_extent,
        )
        if titles is not None:
            ax.set_title(titles[j])
    title = fig.suptitle(f"t = {temporal_grid[0]:.2f}")

    def animate(i):
        for j, ax in enumerate(ax_s.flatten()):
            ax.clear()
            plot_state_2d(
                trj[j, i],
                vlim=vlim,
                cmap=cmap,
                ax=ax,
            )
            if titles is not None:
                ax.set_title(titles[j])
        title.set_text(f"t = {temporal_grid[i]:.2f}")

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trj.shape[1], interval=100, blit=False)

    return ani


def animate_state_3d_facet(
    trj: Union[Float[Array, "T C N N N"], Float[Array, "B T 1 N N N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles=None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    bg_color: Union[
        Literal["black"],
        Literal["white"],
        tuple[jnp.int8, jnp.int8, jnp.int8, jnp.int8],
    ] = "white",
    resolution: int = 384,
    cmap: str = "RdBu_r",
    transfer_function: callable = zigzag_alpha,
    distance_scale: float = 10.0,
    gamma_correction: float = 2.4,
    chunk_size: int = 64,
    **kwargs,
):
    """
    Animate a facet of trajectories of 3d states as volume renderings.

    Requires the input to be either a five-axis array or a six-axis array:

    - If `facet_over_channels` is `True`, the input must be a five-axis array
        with a leading time axis, a channel axis, and three spatial axes. Each
        faceted subplot displays a different channel.
    - If `facet_over_channels` is `False`, the input must be a six-axis array
        with a leading batch axis, a time axis, a channel axis, and three spatial
        axes. Each faceted subplot displays a different batch. Only the zeroth
        dimension in the channel axis is plotted.

    **Arguments**:

    - `trj`: The trajectory of states to animate. Must be a five-axis array with
        shape `(n_timesteps, n_channels, n_spatial, n_spatial, n_spatial)` if
        `facet_over_channels` is `True`, or a six-axis array with shape
        `(n_batches, n_timesteps, n_channels, n_spatial, n_spatial, n_spatial)`
        if `facet_over_channels` is `False`.
    - `facet_over_channels`: Whether to facet over the channel axis or the batch
        axis. Default is `True`.
    - `vlim`: The limits of the colorbar. Default is `(-1, 1)`.
    - `grid`: The grid of subplots. Default is `(3, 3)`.
    - `figsize`: The size of the figure. Default is `(10, 10)`.
    - `titles`: The titles for each subplot. Default is `None`.
    - `domain_extent`: The extent of the spatial domain. Default is `None`. This
        affects the x-axis and y-axis limits of the plot.
    - `dt`: The time step between each frame. Default is `None`.
    - `include_init`: Whether to the state starts at an initial condition (t=0)
        or at the first frame in the trajectory. This affects is the the time
        range is [0, (T-1)dt] or [dt, Tdt]. Default is `False`.
    - `bg_color`: The background color. Either `"black"`, `"white"`, or a tuple
        of RGBA values. Default is `"white"`.
    - `resolution`: The resolution of the output image (affects render time).
        Default is `384`.
    - `cmap`: The colormap to use. Default is `"RdBu_r"`.
    - `transfer_function`: The transfer function to use, i.e., how values within
        the `vlim` range are mapped to alpha values. Default is `zigzag_alpha`.
    - `distance_scale`: The distance scale of the volume renderer. Default is
        `10.0`.
    - `gamma_correction`: The gamma correction to apply to the image. Default is
        `2.4`.
    - `chunk_size`: The number of images to render at once. Default is `64`.

    **Returns**:

    - `ani`: The animation object.

    **Note:**

    - This function requires the `vape` volume renderer package.
    """
    if facet_over_channels:
        if trj.ndim != 5:
            raise ValueError("trj must be a five-axis array.")
    else:
        if trj.ndim != 6:
            raise ValueError("trj must be a six-axis array.")

    if facet_over_channels:
        trj = jnp.swapaxes(trj, 0, 1)
        trj = trj[:, :, None]

    trj_wrapped = jax.vmap(jax.vmap(wrap_bc))(trj)

    imgs = []
    for facet_entry_trj in trj_wrapped:
        facet_entry_trj_no_channel = facet_entry_trj[:, 0]
        imgs.append(
            volume_render_state_3d(
                facet_entry_trj_no_channel,
                vlim=vlim,
                bg_color=bg_color,
                resolution=resolution,
                cmap=cmap,
                transfer_function=transfer_function,
                distance_scale=distance_scale,
                gamma_correction=gamma_correction,
                chunk_size=chunk_size,
                **kwargs,
            )
        )

    # shape = (B, T, resolution, resolution, 3)
    imgs = jnp.stack(imgs)

    if include_init:
        temporal_grid = jnp.arange(trj.shape[1])
    else:
        temporal_grid = jnp.arange(1, trj.shape[1] + 1)

    if dt is not None:
        temporal_grid *= dt

    fig, ax_s = plt.subplots(*grid, figsize=figsize)

    # num_subplots = trj.shape[0]

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
    for j, ax in enumerate(ax_s.flatten()):
        ax.imshow(imgs[j, 0])
        ax.axis("off")
        # if j >= num_subplots:
        #     ax.remove()
        # else:
        if titles is not None:
            ax.set_title(titles[j])
    title = fig.suptitle(f"t = {temporal_grid[0]:.2f}")

    def animate(i):
        for j, ax in enumerate(ax_s.flatten()):
            ax.clear()
            ax.imshow(imgs[j, i])
            ax.axis("off")
            if titles is not None:
                ax.set_title(titles[j])
        title.set_text(f"t = {temporal_grid[i]:.2f}")

    ani = FuncAnimation(fig, animate, frames=trj.shape[1], interval=100, blit=False)

    plt.close(fig)

    return ani


def animate_spatio_temporal_2d_facet():
    # TODO
    raise NotImplementedError("Not implemented yet.")
