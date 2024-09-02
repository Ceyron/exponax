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


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def volume_render_state_3d(
    states: Float[Array, "B N N N"],
    *,
    vlim: tuple[float, float] = (-1.0, 1.0),
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
) -> Float[Array, "B resolution resolution 4"]:
    """
    (Batched) rendering using the vape volume renderer.

    **Arguments:**

    - `states`: The states to render, shape `(B, N, N, N)`. To just render one
        image, this array must have a leading singleton axis (i.e., has shape
        `(1, N, N, N)`), then extract the one image from the returned array.
    - `vlim`: The min and max values for the colormap.
    - `bg_color`: The background color. Either `"black"`, `"white"`, or a tuple
        of RGBA values.
    - `resolution`: The resolution of the output image (affects render time).
    - `cmap`: The colormap to use.
    - `transfer_function`: The transfer function to use, i.e., how values within
        the `vlim` range are mapped to alpha values.
    - `distance_scale`: The distance scale of the volume renderer.
    - `gamma_correction`: The gamma correction to apply to the image.
    - `chunk_size`: The number of images to render at once.

    **Returns:**

    - `imgs`: The rendered images, in terms of RBGA-images (channels-last) and a
        leading batch axis, shape `(B, resolution, resolution, 4)`.
    """
    if states.ndim != 4:
        raise ValueError("state must be a four-axis array.")
    try:
        import vape4d
    except ImportError:
        raise ImportError(
            """
                          This function requires the `vape4d` volume renderer package.
                            Please install it via `pip install vape4d`.
                          """
        )

    if bg_color == "black":
        bg_color = (0, 0, 0, 255)
    elif bg_color == "white":
        bg_color = (255, 255, 255, 255)

    # Need to convert to numpy array
    states = np.array(states).astype(np.float32)

    cmap_with_alpha_transfer = transfer_function(plt.get_cmap(cmap))

    num_images = states.shape[0]

    imgs = []
    for time_steps in chunk_list(range(num_images), chunk_size):
        if num_images == 1:
            sub_time_steps = [0.0]
        else:
            sub_time_steps = [i / (num_images - 1) for i in time_steps]
        imgs_this_batch = vape4d.render(
            states,
            cmap=cmap_with_alpha_transfer,
            time=sub_time_steps,
            width=resolution,
            height=resolution,
            background=bg_color,
            vmin=vlim[0],
            vmax=vlim[1],
            distance_scale=distance_scale,
        )
        if num_images == 1:
            imgs_this_batch = imgs_this_batch[None]
        imgs.append(imgs_this_batch)

    imgs = np.concatenate(imgs, axis=0)
    imgs = ((imgs / 255.0) ** (gamma_correction) * 255).astype(np.uint8)

    return imgs
