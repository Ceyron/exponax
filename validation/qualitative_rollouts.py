"""
Work in Progress.
"""

import os
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(".")
import exponax as ex  # noqa: E402

ic_key = jax.random.PRNGKey(0)

CONFIGURATIONS_1D = [
    (
        ex.stepper.Advection(1, 3.0, 110, 0.01, velocity=0.3),
        "advection",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(1, 3.0, 110, 0.01, diffusivity=0.01),
        "diffusion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        100,
        (-1.0, 1.0),
    ),
]

p_meter = tqdm(CONFIGURATIONS_1D, desc="", total=len(CONFIGURATIONS_1D))
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
img_folder = dir_path / Path("qualitative_rollouts")
img_folder.mkdir(exist_ok=True)


# 1d problems (produce spatio-temporal plots)
for stepper_1d, name, ic_distribution, steps, vlim in CONFIGURATIONS_1D:
    p_meter.set_description(f"1d {name}")

    ic = ic_distribution(stepper_1d.num_points, key=ic_key)
    trj = ex.rollout(stepper_1d, steps, include_init=True)(ic)
    num_channels = stepper_1d.num_channels
    fig, ax_s = plt.subplots(num_channels, 1, figsize=(5, 5 * num_channels))
    if num_channels == 1:
        ax_s = [
            ax_s,
        ]
    for i, ax in enumerate(ax_s):
        ax.imshow(
            trj[:, i, :].T,
            aspect="auto",
            origin="lower",
            vmin=vlim[0],
            vmax=vlim[1],
            cmap="RdBu_r",
        )
        ax.set_title(f"{name} channel {i}")
        ax.set_xlabel("time")
        ax.set_ylabel("space")

    fig.savefig(img_folder / f"{name}_1d.png")
    plt.close(fig)

    p_meter.update(1)
