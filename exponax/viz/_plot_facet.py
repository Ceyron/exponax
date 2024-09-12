from typing import Literal, TypeVar, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float

from ._plot import (
    plot_spatio_temporal,
    plot_spatio_temporal_2d,
    plot_state_1d,
    plot_state_2d,
    plot_state_3d,
)
from ._volume import zigzag_alpha

N = TypeVar("N")


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

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
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

    plt.close(fig)

    return fig


def plot_spatio_temporal_facet(
    trjs: Union[Float[Array, "T C N"], Float[Array, "B T 1 N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    cmap: str = "RdBu_r",
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles: list[str] = None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    """
    Plot a facet of spatio-temporal trajectories.

    Requires the input to be a real array with either three or four axes:

    * Three axes: a leading time axis, a channel axis, and a spatial axis. The
        faceting is performed over the channel axis. Requires the
        `facet_over_channels` argument to be `True` (default).
    * Four axes: a leading batch axis, a time axis, a channel axis, and a
      spatial
        axis. The faceting is performed over the batch axis. Requires the
        `facet_over_channels` argument to be `False`. Only the zeroth channel
        for each trajectory will be plotted.

    Periodic boundary conditions will be applied to the spatial axis (the state
    is wrapped around).

    **Arguments:**

    - `trjs`: The trajectories to plot as a three or four axis array. See above
        for the requirements.
    - `facet_over_channels`: Whether to facet over the channel axis (three axes)
        or the batch axis (four axes).
    - `vlim`: The limits of the color scale.
    - `cmap`: The colormap to use.
    - `grid`: The grid layout for the facet plot. This should be a tuple with
        two integers. If the number of trajectories is less than the product of
        the grid, the remaining axes will be removed.
    - `figsize`: The size of the figure.
    - `titles`: The titles for each plot. This should be a list of strings with
        the same length as the number of trajectories.
    - `domain_extent`: The extent of the spatial domain. If not provided, the
        domain extent will be the number of points in the spatial axis. This
        adjusts the y-axis.
    - `dt`: The time step. This adjust the extent of the x-axis. If not
        provided, the time axis will be the number of time steps.
    - `include_init`: Will affect the ticks of the time axis. If `True`, they
        will start at zero. If `False`, they will start at the time step.
    - `**kwargs`: Additional arguments to pass to the imshow function.

    **Returns:**

    - The figure.
    """
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

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
    for i, ax in enumerate(ax_s.flatten()):
        single_trj = trjs[i]
        plot_spatio_temporal(
            single_trj,
            vlim=vlim,
            cmap=cmap,
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

    plt.close(fig)

    return fig


def plot_state_2d_facet(
    states: Union[Float[Array, "C N N"], Float[Array, "B 1 N N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    cmap: str = "RdBu_r",
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles: list[str] = None,
    domain_extent: float = None,
    **kwargs,
):
    """
    Plot a facet of 2d states.

    Requires the input to be a real array with three or four axes:

    * Three axes: a leading channel axis, and two subsequent spatial axes. The
        facet will be done over the channel axis, requires the
        `facet_over_channels` argument to be `True` (default).
    * Four axes: a leading batch axis, a channel axis, and two subsequent
        spatial axes. The facet will be done over the batch axis, requires the
        `facet_over_channels` argument to be `False`. Only the zeroth channel
        for each state will be plotted.

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments:**

    - `states`: The states to plot as a three or four axis array. See above for
        the requirements.
    - `facet_over_channels`: Whether to facet over the channel axis (three axes)
        or the batch axis (four axes).
    - `vlim`: The limits of the color scale.
    - `cmap`: The colormap to use.
    - `grid`: The grid layout for the facet plot. This should be a tuple with
        two integers. If the number of states is less than the product of the
        grid, the remaining axes will be removed.
    - `figsize`: The size of the figure.
    - `titles`: The titles for each plot. This should be a list of strings with
        the same length as the number of states.
    - `domain_extent`: The extent of the spatial domain. If not provided, the
        domain extent will be the number of points in the spatial axes. This
        adjusts the x and y axes.
    - `**kwargs`: Additional arguments to pass to the imshow function.

    **Returns:**

    - The figure.
    """
    if facet_over_channels:
        if states.ndim != 3:
            raise ValueError("states must be a three-axis array.")
        states = states[:, None, :, :]
    else:
        if states.ndim != 4:
            raise ValueError("states must be a four-axis array.")

    fig, ax_s = plt.subplots(*grid, sharex=True, sharey=True, figsize=figsize)

    num_subplots = states.shape[0]

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
    for i, ax in enumerate(ax_s.flatten()):
        plot_state_2d(
            states[i],
            vlim=vlim,
            cmap=cmap,
            ax=ax,
            domain_extent=domain_extent,
            **kwargs,
        )
        if i >= num_subplots:
            ax.remove()
        else:
            if titles is not None:
                ax.set_title(titles[i])

    plt.close(fig)

    return fig


def plot_state_3d_facet(
    states: Union[Float[Array, "C N N N"], Float[Array, "B 1 N N N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles: list[str] = None,
    domain_extent: float = None,
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
    **kwargs,
):
    """
    Plot a facet of 3d states as volume renders.

    Requires the input to be a real array with four or five axes: a leading
    batch axis, a channel axis, and three subsequent spatial axes. The facet
    will be done over the batch axis, requires the `facet_over_channels`
    argument to be `False`. Only the zeroth channel for each state will be
    plotted.

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments:**

    - `states`: The states to plot as a four or five axis array. See above for
        the requirements.
    - `facet_over_channels`: Whether to facet over the channel axis (four axes)
        or the batch axis (five axes).
    - `vlim`: The limits of the color scale.
    - `grid`: The grid layout for the facet plot. This should be a tuple with
        two integers. If the number of states is less than the product of the
        grid, the remaining axes will be removed.
    - `figsize`: The size of the figure.
    - `titles`: The titles for each plot. This should be a list of strings with
        the same length as the number of states.
    - `domain_extent`: (Unused as of now)
    - `bg_color`: The background color. Either `"black"`, `"white"`, or a tuple
        of RGBA values.
    - `resolution`: The resolution of the output image (affects render time).
    - `cmap`: The colormap to use.
    - `transfer_function`: The transfer function to use, i.e., how values within
        the `vlim` range are mapped to alpha values.
    - `distance_scale`: The distance scale of the volume renderer.
    - `gamma_correction`: The gamma correction to apply to the image.

    **Returns:**

    - The figure.

    **Note:**

    - This function requires the `vape` volume renderer package.
    """
    if facet_over_channels:
        if states.ndim != 4:
            raise ValueError("states must be a four-axis array.")
        states = states[:, None, :, :, :]
    else:
        if states.ndim != 5:
            raise ValueError("states must be a five-axis array.")

    fig, ax_s = plt.subplots(*grid, figsize=figsize)

    num_subplots = states.shape[0]

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
    for i, ax in enumerate(ax_s.flatten()):
        plot_state_3d(
            states[i],
            vlim=vlim,
            domain_extent=domain_extent,
            ax=ax,
            bg_color=bg_color,
            resolution=resolution,
            cmap=cmap,
            transfer_function=transfer_function,
            distance_scale=distance_scale,
            gamma_correction=gamma_correction,
            **kwargs,
        )
        if i >= num_subplots:
            ax.remove()
        else:
            if titles is not None:
                ax.set_title(titles[i])

    plt.close(fig)

    return fig


def plot_spatio_temporal_2d_facet(
    trjs: Union[Float[Array, "T C N N"], Float[Array, "B T 1 N N"]],
    *,
    facet_over_channels: bool = True,
    vlim: tuple[float, float] = (-1.0, 1.0),
    grid: tuple[int, int] = (3, 3),
    figsize: tuple[float, float] = (10, 10),
    titles: list[str] = None,
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
    **kwargs,
):
    """
    Plot a facet of spatio-temporal trajectories.

    Requires the input to be a real array with either four or five axes:

    * Four axes: a leading time axis, a channel axis, and two subsequent spatial
        axes. The faceting is performed over the channel axis. Requires the
        `facet_over_channels` argument to be `True` (default).
    * Five axes: a leading batch axis, a time axis, a channel axis, and two
        subsequent spatial axes. The faceting is performed over the batch axis.
        Requires the `facet_over_channels` argument to be `False`.

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments:**

    - `trjs`: The trajectories to plot as a four or five axis array. See above
        for the requirements.
    - `facet_over_channels`: Whether to facet over the channel axis (four axes)
        or the batch axis (five axes).
    - `vlim`: The limits of the color scale.
    - `grid`: The grid layout for the facet plot. This should be a tuple with
        two integers. If the number of trajectories is less than the product of
        the grid, the remaining axes will be removed.
    - `figsize`: The size of the figure.
    - `titles`: The titles for each plot. This should be a list of strings with
        the same length as the number of trajectories.
    - `domain_extent`: (Unused as of now)
    - `dt`: (Unused as of now)
    - `include_init`: (Unused as of now)
    - `bg_color`: The background color. Either `"black"`, `"white"`, or a tuple
        of RGBA values.
    - `resolution`: The resolution of the output image (affects render time).
    - `cmap`: The colormap to use.
    - `transfer_function`: The transfer function to use, i.e., how values within
        the `vlim` range are mapped to alpha values.
    - `distance_scale`: The distance scale of the volume renderer.
    - `gamma_correction`: The gamma correction to apply to the image.

    **Returns:**

    - The figure.

    **Note:**

    - This function requires the `vape` volume renderer package.
    """
    if facet_over_channels:
        if trjs.ndim != 4:
            raise ValueError("trjs must be a four-axis array.")
    else:
        if trjs.ndim != 5:
            raise ValueError("trjs must be a five-axis array.")

    fig, ax_s = plt.subplots(*grid, figsize=figsize)

    if facet_over_channels:
        trjs = jnp.swapaxes(trjs, 0, 1)
        trjs = trjs[:, :, None, :, :]

    num_subplots = trjs.shape[0]

    if grid[0] * grid[1] == 1:
        ax_s = np.array([[ax_s]])
    for i, ax in enumerate(ax_s.flatten()):
        single_trj = trjs[i]
        plot_spatio_temporal_2d(
            single_trj,
            vlim=vlim,
            ax=ax,
            domain_extent=domain_extent,
            dt=dt,
            include_init=include_init,
            bg_color=bg_color,
            resolution=resolution,
            cmap=cmap,
            transfer_function=transfer_function,
            distance_scale=distance_scale,
            gamma_correction=gamma_correction,
            **kwargs,
        )
        if i >= num_subplots:
            ax.remove()
        else:
            if titles is not None:
                ax.set_title(titles[i])

    plt.close(fig)

    return fig
