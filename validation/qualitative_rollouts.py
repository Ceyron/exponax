"""
Work in Progress.
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

sys.path.append(".")
import exponax as ex  # noqa: E402

ic_key = jax.random.PRNGKey(0)

CONFIGURATIONS_1D = [
    # Linear
    (
        ex.stepper.Advection(1, 3.0, 110, 0.01, velocity=0.3),
        "advection",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(1, 3.0, 110, 0.01, diffusivity=0.01),
        "diffusion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.AdvectionDiffusion(
            1, 3.0, 110, 0.01, diffusivity=0.01, velocity=0.3
        ),
        "advection_diffusion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Dispersion(1, 3.0, 110, 0.01, dispersivity=0.01),
        "dispersion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=3),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.HyperDiffusion(1, 3.0, 110, 0.01, hyper_diffusivity=0.001),
        "hyper_diffusion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.GeneralLinearStepper(
            1,
            3.0,
            110,
            0.01,
            coefficients=[0.0, 0.0, 0.1, 0.0001],
        ),
        "dispersion_diffusion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.GeneralLinearStepper(
            1,
            3.0,
            110,
            0.01,
            coefficients=[0.0, 0.0, 0.0, 0.0001, -0.001],
        ),
        "dispersion_hyper_diffusion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    # Nonlinear
    (
        ex.stepper.Burgers(1, 3.0, 110, 0.01, diffusivity=0.03),
        "burgers",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.KortewegDeVries(1, 20.0, 110, 0.01),
        "kdv",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-2.0, 2.0),
    ),
    (
        ex.stepper.KuramotoSivashinsky(1, 60.0, 110, 0.5),
        "ks",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=3),
        500,
        200,
        (-6.5, 6.5),
    ),
    (
        ex.stepper.KuramotoSivashinskyConservative(1, 60.0, 110, 0.5),
        "ks_conservative",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=3),
        500,
        200,
        (-2.5, 2.5),
    ),
    # Reaction
    (
        ex.reaction.FisherKPP(1, 10.0, 256, 0.001, reactivity=10.0),
        "fisher_kpp",
        ex.ic.ClampingICGenerator(
            ex.ic.RandomTruncatedFourierSeries(1, cutoff=5), (0.0, 1.0)
        ),
        0,
        300,
        (-1.0, 1.0),
    ),
]

CONFIGURATIONS_2D = [
    # Linear
    (
        ex.stepper.Advection(2, 3.0, 75, 0.1, velocity=jnp.array([0.3, -0.5])),
        "advection",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(2, 3.0, 75, 0.1, diffusivity=0.01),
        "diffusion",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(2, 3.0, 75, 0.1, diffusivity=jnp.array([0.01, 0.05])),
        "diffusion_diagonal",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(
            2, 3.0, 75, 0.1, diffusivity=jnp.array([[0.02, 0.01], [0.01, 0.05]])
        ),
        "diffusion_anisotropic",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.AdvectionDiffusion(
            2, 3.0, 75, 0.1, diffusivity=0.01, velocity=jnp.array([0.3, -0.5])
        ),
        "advection_diffusion",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Dispersion(2, 3.0, 75, 0.1, dispersivity=0.01),
        "dispersion",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=3),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Dispersion(
            2, 3.0, 75, 0.1, dispersivity=0.01, advect_on_diffusion=True
        ),
        "dispersion_advect_on_diffuse",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=3),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.HyperDiffusion(2, 3.0, 75, 0.1, hyper_diffusivity=0.0001),
        "hyper_diffusion",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.HyperDiffusion(
            2, 3.0, 75, 0.1, hyper_diffusivity=0.0001, diffuse_on_diffuse=True
        ),
        "hyper_diffusion_diffuse_on_diffuse",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    # Nonlinear
    (
        ex.stepper.Burgers(2, 3.0, 65, 0.05, diffusivity=0.02),
        "burgers",
        ex.ic.RandomMultiChannelICGenerator(
            2
            * [
                ex.ic.ClampingICGenerator(
                    ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
                    (-1.0, 1.0),
                ),
            ]
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Burgers(2, 3.0, 65, 0.05, diffusivity=0.02, single_channel=True),
        "burgers_single_channel",
        ex.ic.ClampingICGenerator(
            ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
            (-1.0, 1.0),
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.KortewegDeVries(2, 20.0, 65, dt=0.01),
        "kdv",
        ex.ic.RandomMultiChannelICGenerator(
            2
            * [
                ex.ic.ClampingICGenerator(
                    ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
                    (-1.0, 1.0),
                ),
            ]
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.KortewegDeVries(2, 20.0, 65, dt=0.01, single_channel=True),
        "kdv_single_channel",
        ex.ic.ClampingICGenerator(
            ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
            (-1.0, 1.0),
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.KuramotoSivashinsky(2, 30.0, 60, 0.1),
        "ks",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=3),
        500,
        100,
        (-6.5, 6.5),
    ),
    (
        ex.RepeatedStepper(
            ex.stepper.NavierStokesVorticity(
                2,
                1.0,
                48,
                0.1 / 10,
                diffusivity=0.0003,
            ),
            10,
        ),
        "ns_vorticity",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5),
        0,
        100,
        (-5.0, 5.0),
    ),
    (
        ex.RepeatedStepper(
            ex.stepper.KolmogorovFlowVorticity(
                2,
                2 * jnp.pi,
                72,
                1.0 / 50,
                diffusivity=0.001,
            ),
            50,
        ),
        "kolmogorov_vorticity",
        ex.ic.DiffusedNoise(2, zero_mean=True),
        200,
        100,
        (-5.0, 5.0),
    ),
    # Reaction
    (
        ex.reaction.CahnHilliard(2, 128, 300, 0.001, gamma=1e-3),
        "cahn_hilliard",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=10),
        0,
        100,
        (-10.0, 10.0),
    ),
    (
        ex.reaction.GrayScott(2, 2.0, 60, 1.0),
        "gray_scott",
        ex.ic.RandomMultiChannelICGenerator(
            [
                ex.ic.RandomGaussianBlobs(2, one_complement=True),
                ex.ic.RandomGaussianBlobs(2),
            ]
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.reaction.SwiftHohenberg(2, 20.0 * jnp.pi, 100, 0.1),
        "swift_hohenberg",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5, max_one=True),
        0,
        100,
        (-1.0, 1.0),
    ),
]

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
img_folder = dir_path / Path("qualitative_rollouts")
img_folder.mkdir(exist_ok=True)


p_meter_1d = tqdm(CONFIGURATIONS_1D, desc="", total=len(CONFIGURATIONS_1D))
# 1d problems (produce spatio-temporal plots)
for stepper_1d, name, ic_distribution, warmup_steps, steps, vlim in CONFIGURATIONS_1D:
    p_meter_1d.set_description(f"1d {name}")

    ic = ic_distribution(stepper_1d.num_points, key=ic_key)
    ic = ex.repeat(stepper_1d, warmup_steps)(ic)
    trj = ex.rollout(stepper_1d, steps, include_init=True)(ic)
    jnp.save(img_folder / f"{name}_1d.npy", trj)

    num_channels = stepper_1d.num_channels
    fig, ax_s = plt.subplots(num_channels, 1, figsize=(8, 4 * num_channels))
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

    p_meter_1d.update(1)

p_meter_1d.close()

p_meter_2d = tqdm(CONFIGURATIONS_2D, desc="", total=len(CONFIGURATIONS_2D))
# 2d problems (produce animations)
for stepper_2d, name, ic_distribution, warmup_steps, steps, vlim in CONFIGURATIONS_2D:
    p_meter_2d.set_description(f"2d {name}")

    ic = ic_distribution(stepper_2d.num_points, key=ic_key)
    ic = ex.repeat(stepper_2d, warmup_steps)(ic)
    trj = ex.rollout(stepper_2d, steps, include_init=True)(ic)
    jnp.save(img_folder / f"{name}_2d.npy", trj)

    num_channels = stepper_2d.num_channels
    fig, ax_s = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))
    if num_channels == 1:
        ax_s = [
            ax_s,
        ]
    im_s = []
    for i, ax in enumerate(ax_s):
        im = ax.imshow(
            trj[0, i, :, :].T,
            aspect="auto",
            origin="lower",
            vmin=vlim[0],
            vmax=vlim[1],
            cmap="RdBu_r",
        )
        im_s.append(im)
        ax.set_title(f"{name} channel {i}")
        ax.set_xlabel("time")
        ax.set_ylabel("space")
    fig.suptitle(f"{name} 2d, t_i = 0")

    def animate(i):
        for j, im in enumerate(im_s):
            im.set_data(trj[i, j, :, :].T)
        fig.suptitle(f"{name} 2d, t_i = {i:04d}")
        return im_s

    plt.close(fig)

    ani = FuncAnimation(fig, animate, frames=trj.shape[0], interval=100, blit=False)

    ani.save(img_folder / f"{name}_2d.mp4")
    del ani

    p_meter_2d.update(1)

p_meter_2d.close()
