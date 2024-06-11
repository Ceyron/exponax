"""
High-Level abstractions around the vape volume renderer.
"""

import copy
from typing import Literal, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def triangle_wave(x, p):
    return 2 * jnp.abs(x / p - jnp.floor(x / p + 0.5))


def zigzag_alpha(cmap, min_alpha=0.0):
    """changes the alpha channel of a colormap to be linear (0->0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:a
        Colormap: new colormap
    """
    if isinstance(cmap, ListedColormap):
        colors = copy.deepcopy(cmap.colors)
        for i, a in enumerate(colors):
            a.append(
                (triangle_wave(i / (cmap.N - 1), 0.5) * (1 - min_alpha)) + min_alpha
            )
        return ListedColormap(colors, cmap.name)
    elif isinstance(cmap, LinearSegmentedColormap):
        segmentdata = copy.deepcopy(cmap._segmentdata)
        segmentdata["alpha"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 1.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.75, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        return LinearSegmentedColormap(cmap.name, segmentdata)
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )


def render_3d_state(
    state: Float[Array, "1 N N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
    domain_extent: float = None,
    ax=None,
    bg_color: Union[
        Literal["black"],
        Literal["white"],
        tuple[jnp.int8, jnp.int8, jnp.int8, jnp.int8],
    ] = "black",
    resolution: int = 384,
    cmap: str = "RdBu_r",
    transfer_function: callable = zigzag_alpha,
    distance_scale: float = 10.0,
    **kwargs,
):
    if state.ndim != 4:
        raise ValueError("state must be a four-axis array.")
    try:
        import vape
    except ImportError:
        raise ImportError("This function requires the `vape` volume renderer package.")

    if bg_color == "black":
        bg_color = (0, 0, 0, 255)
    elif bg_color == "white":
        bg_color = (255, 255, 255, 255)

    # Need to convert to numpy array
    state = np.array(state).astype(np.float32)

    cmap_with_alpha_transfer = transfer_function(plt.get_cmap(cmap))

    imgs = vape.render(
        state,
        cmap=cmap_with_alpha_transfer,
        time=[
            0.0,
        ],
        width=resolution,
        height=resolution,
        background=bg_color,
        vmin=vlim[0],
        vmax=vlim[1],
        distance_scale=distance_scale,
    )

    img = imgs[0]

    return img
