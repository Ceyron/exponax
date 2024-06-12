from typing import Literal, TypeVar, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float

from .._utils import make_grid, wrap_bc
from ._volume import volume_render_state_3d, zigzag_alpha

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

    - If `ax` is not provided, returns the figure. Otherwise, returns the plot
        object.
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
        return_all = True
        fig, ax = plt.subplots()
    else:
        return_all = False

    p = ax.plot(grid[0], state_wrapped.T, label=labels, **kwargs)
    ax.set_ylim(vlim)
    ax.grid()
    if labels is not None:
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if return_all:
        plt.close(fig)
        return fig
    else:
        return p


def plot_spatio_temporal(
    trj: Float[Array, "T 1 N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    cmap: str = "RdBu_r",
    ax=None,
    domain_extent: float = None,
    dt: float = None,
    include_init: bool = False,
    **kwargs,
):
    """
    Plot a trajectory of a 1d state as a spatio-temporal plot (space in y-axis,
    and time in x-axis).

    Requires the input to be a real array with three axis: a leading time axis,
    a channel axis, and a spatial axis. Only the leading dimension in the
    channel axis will be plotted. See `plot_spatio_temporal_facet` for plotting
    multiple trajectories.

    Periodic boundary conditions will be applied to the spatial axis (the state
    is wrapped around).

    **Arguments:**

    - `trj`: The trajectory to plot as a three axis array. The first axis should
        be the time axis, the second axis the channel axis, and the third axis
        the spatial axis.
    - `vlim`: The limits of the color scale.
    - `cmap`: The colormap to use.
    - `ax`: The axis to plot on. If not provided, a new figure will be created.
    - `domain_extent`: The extent of the spatial domain. If not provided, the
        domain extent will be the number of points in the spatial axis. This
        adjusts the y-axis.
    - `dt`: The time step. This adjust the extent of the x-axis. If not
        provided, the time axis will be the number of time steps.
    - `include_init`: Will affect the ticks of the time axis. If `True`, they
        will start at zero. If `False`, they will start at the time step.
    - `**kwargs`: Additional arguments to pass to the imshow function.

    **Returns:**

    - If `ax` is not provided, returns the figure. Otherwise, returns the image
        object.
    """
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
        return_all = True
    else:
        return_all = False

    im = ax.imshow(
        trj_wrapped[:, 0, :].T,
        vmin=vlim[0],
        vmax=vlim[1],
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=(*time_range, *space_range),
        **kwargs,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Space")

    if return_all:
        plt.close(fig)
        return fig
    else:
        return im


def plot_state_2d(
    state: Float[Array, "1 N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    cmap: str = "RdBu_r",
    domain_extent: float = None,
    ax=None,
    **kwargs,
):
    """
    Visualizes a two-dimensional state as an image.

    Requires the input to be a real array with three axes: a leading channel
    axis, and two subsequent spatial axes. This function will visualize the
    zeroth channel. For plotting multiple channels at the same time, see
    `plot_state_2d_facet`.

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments:**

    - `state`: The state to plot as a three axis array. The first axis should be
        the channel axis, and the subsequent two axes the spatial axes.
    - `vlim`: The limits of the color scale.
    - `cmap`: The colormap to use.
    - `domain_extent`: The extent of the spatial domain. If not provided, the
        domain extent will be the number of points in the spatial axes. This
        adjusts the x and y axes.
    - `ax`: The axis to plot on. If not provided, a new figure will be created.
    - `**kwargs`: Additional arguments to pass to the imshow function.

    **Returns:**

    - If `ax` is not provided, returns the figure. Otherwise, returns the image
        object.
    """
    if state.ndim != 3:
        raise ValueError("state must be a three-axis array.")

    if domain_extent is not None:
        space_range = (0, domain_extent)
    else:
        # One more because we wrapped the BC
        space_range = (0, state.shape[-1])

    state_wrapped = wrap_bc(state)

    if ax is None:
        fig, ax = plt.subplots()
        return_all = True
    else:
        return_all = False

    im = ax.imshow(
        state_wrapped.T,
        vmin=vlim[0],
        vmax=vlim[1],
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=(*space_range, *space_range),
        **kwargs,
    )
    ax.set_xlabel("x_0")
    ax.set_ylabel("x_1")
    ax.set_aspect("equal")

    if return_all:
        plt.close(fig)
        return fig
    else:
        return im


def plot_state_3d(
    state: Float[Array, "1 N N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    ax=None,
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
    Visualizes a three-dimensional state as a volume rendering.

    Requires the input to be a real array with four axes: a leading channel axis,
    and three subsequent spatial axes. This function will visualize the zeroth
    channel. For plotting multiple channels at the same time, see
    `plot_state_3d_facet`.

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments:**

    - `state`: The state to plot as a four axis array. The first axis should be
        the channel axis, and the subsequent three axes the spatial axes.
    - `vlim`: The limits of the color scale.
    - `domain_extent`: (Unused as of now)
    - `ax`: The axis to plot on. If not provided, a new figure will be created.
    - `bg_color`: The background color. Either `"black"`, `"white"`, or a tuple
        of RGBA values.
    - `resolution`: The resolution of the output image (affects render time).
    - `cmap`: The colormap to use.
    - `transfer_function`: The transfer function to use, i.e., how values within
        the `vlim` range are mapped to alpha values.
    - `distance_scale`: The distance scale of the volume renderer.
    - `gamma_correction`: The gamma correction to apply to the image.

    **Returns:**

    - If `ax` is not provided, returns the figure. Otherwise, returns the image
        object.

    **Note:**

    - This function requires the `vape` volume renderer package.
    """
    if state.ndim != 4:
        raise ValueError("state must be a four-axis array.")

    one_channel_state = state[0:1]
    one_channel_state_wrapped = wrap_bc(one_channel_state)

    imgs = volume_render_state_3d(
        one_channel_state_wrapped,
        vlim=vlim,
        bg_color=bg_color,
        resolution=resolution,
        cmap=cmap,
        transfer_function=transfer_function,
        distance_scale=distance_scale,
        gamma_correction=gamma_correction,
        **kwargs,
    )

    img = imgs[0]

    if ax is None:
        fig, ax = plt.subplots()
        return_all = True
    else:
        return_all = False

    im = ax.imshow(img)
    ax.axis("off")

    if return_all:
        plt.close(fig)
        return fig
    else:
        return im


def plot_spatio_temporal_2d(
    trj: Float[Array, "T 1 N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    ax=None,
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
    Plot a trajectory of a 2d state as a spatio-temporal plot visualized by a
    volume render (space in in plain parallel to screen, and time in the
    direction into the screen).

    Requires the input to be a real array with four axes: a leading time axis, a
    channel axis, and two subsequent spatial axes. Only the leading dimension in
    the channel axis will be plotted. See `plot_spatio_temporal_facet` for
    plotting multiple trajectories (e.g. for problems consisting of multiple
    channels like Burgers simulations).

    Periodic boundary conditions will be applied to the spatial axes (the state
    is wrapped around).

    **Arguments:**

    - `trj`: The trajectory to plot as a four axis array. The first axis should
        be the time axis, the second axis the channel axis, and the third and
        fourth axes the spatial axes.
    - `vlim`: The limits of the color scale.
    - `ax`: The axis to plot on. If not provided, a new figure will be created.
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

    - If `ax` is not provided, returns the figure. Otherwise, returns the image
        object.

    **Note:**

    - This function requires the `vape` volume renderer package.
    """
    if trj.ndim != 4:
        raise ValueError("trj must be a four-axis array.")

    trj_one_channel = trj[:, 0:1]
    trj_one_channel_wrapped = jax.vmap(wrap_bc)(trj_one_channel)

    trj_reshaped_to_3d = jnp.flip(
        jnp.array(trj_one_channel_wrapped.transpose(1, 2, 3, 0)), 3
    )

    imgs = volume_render_state_3d(
        trj_reshaped_to_3d,
        vlim=vlim,
        bg_color=bg_color,
        resolution=resolution,
        cmap=cmap,
        transfer_function=transfer_function,
        distance_scale=distance_scale,
        gamma_correction=gamma_correction,
        **kwargs,
    )

    img = imgs[0]

    if ax is None:
        fig, ax = plt.subplots()
        return_all = True
    else:
        return_all = False

    im = ax.imshow(img)
    ax.axis("off")

    if return_all:
        plt.close(fig)
        return fig
    else:
        return im
