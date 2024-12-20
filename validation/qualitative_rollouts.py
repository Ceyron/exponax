# Find the dump at: https://github.com/Ceyron/exponax_qualitative_rollouts

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(".")
import exponax as ex  # noqa: E402

ic_key = jax.random.PRNGKey(0)

HAS_VAPE = True  # Set to False if you are on a non-GPU machine to not produce the 3D animations

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
        ex.stepper.generic.GeneralLinearStepper(
            1,
            3.0,
            110,
            0.01,
            linear_coefficients=[0.0, 0.0, 0.1, 0.0001],
        ),
        "dispersion_diffusion",
        ex.ic.RandomTruncatedFourierSeries(1, cutoff=5),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.generic.GeneralLinearStepper(
            1,
            3.0,
            110,
            0.01,
            linear_coefficients=[0.0, 0.0, 0.0, 0.0001, -0.001],
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
        ex.stepper.reaction.FisherKPP(1, 10.0, 256, 0.001, reactivity=10.0),
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
                diffusivity=0.01,
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
        ex.stepper.reaction.CahnHilliard(2, 128, 300, 0.001, gamma=1e-3),
        "cahn_hilliard",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=10),
        0,
        100,
        (-10.0, 10.0),
    ),
    (
        ex.stepper.reaction.GrayScott(2, 2.0, 60, 1.0),
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
        ex.stepper.reaction.SwiftHohenberg(2, 20.0 * jnp.pi, 100, 0.1),
        "swift_hohenberg",
        ex.ic.RandomTruncatedFourierSeries(2, cutoff=5, max_one=True),
        0,
        100,
        (-1.0, 1.0),
    ),
]

CONFIGURATIONS_3D = [
    # Linear
    (
        ex.stepper.Advection(3, 3.0, 32, 0.1, velocity=jnp.array([0.3, -0.5, 0.1])),
        "advection",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(3, 3.0, 32, 0.1, diffusivity=0.01),
        "diffusion",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(
            3, 3.0, 32, 0.1, diffusivity=jnp.array([0.01, 0.05, 0.005])
        ),
        "diffusion_diagonal",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Diffusion(
            3,
            3.0,
            32,
            0.1,
            diffusivity=jnp.array(
                [[0.02, 0.01, 0.005], [0.01, 0.05, 0.01], [0.005, 0.01, 0.03]]
            ),
        ),
        "diffusion_anisotropic",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.AdvectionDiffusion(
            3, 3.0, 32, 0.1, diffusivity=0.01, velocity=jnp.array([0.3, -0.5, 0.1])
        ),
        "advection_diffusion",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Dispersion(3, 3.0, 32, 0.1, dispersivity=0.01),
        "dispersion",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=3),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Dispersion(
            3, 3.0, 32, 0.1, dispersivity=0.01, advect_on_diffusion=True
        ),
        "dispersion_advect_on_diffuse",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=3),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.HyperDiffusion(3, 3.0, 32, 0.1, hyper_diffusivity=0.0001),
        "hyper_diffusion",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.HyperDiffusion(
            3, 3.0, 32, 0.1, hyper_diffusivity=0.0001, diffuse_on_diffuse=True
        ),
        "hyper_diffusion_diffuse_on_diffuse",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
        0,
        30,
        (-1.0, 1.0),
    ),
    # Nonlinear
    (
        ex.stepper.Burgers(3, 3.0, 48, 0.05, diffusivity=0.02),
        "burgers",
        ex.ic.RandomMultiChannelICGenerator(
            3
            * [
                ex.ic.ClampingICGenerator(
                    ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
                    (-1.0, 1.0),
                ),
            ]
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.Burgers(3, 3.0, 48, 0.05, diffusivity=0.01, single_channel=True),
        "burgers_single_channel",
        ex.ic.ClampingICGenerator(
            ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
            (-1.0, 1.0),
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.KortewegDeVries(3, 20.0, 48, dt=0.01),
        "kdv",
        ex.ic.RandomMultiChannelICGenerator(
            3
            * [
                ex.ic.ClampingICGenerator(
                    ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
                    (-1.0, 1.0),
                ),
            ]
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.KortewegDeVries(3, 20.0, 48, dt=0.01, single_channel=True),
        "kdv_single_channel",
        ex.ic.ClampingICGenerator(
            ex.ic.RandomTruncatedFourierSeries(3, cutoff=5),
            (-1.0, 1.0),
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.KuramotoSivashinsky(3, 30.0, 48, 0.1),
        "ks",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=3),
        500,
        100,
        (-6.5, 6.5),
    ),
    (
        ex.stepper.reaction.GrayScott(3, 2.0, 48, 1.0),
        "gray_scott",
        ex.ic.RandomMultiChannelICGenerator(
            [
                ex.ic.RandomGaussianBlobs(3, one_complement=True),
                ex.ic.RandomGaussianBlobs(3),
            ]
        ),
        0,
        100,
        (-1.0, 1.0),
    ),
    (
        ex.stepper.reaction.SwiftHohenberg(3, 20.0 * jnp.pi, 48, 0.1),
        "swift_hohenberg",
        ex.ic.RandomTruncatedFourierSeries(3, cutoff=5, max_one=True),
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
    fig = ex.viz.plot_spatio_temporal_facet(
        trj,
        vlim=vlim,
        titles=[f"{name} channel {i}" for i in range(num_channels)],
        grid=(num_channels, 1),
        figsize=(8, 4 * num_channels),
    )

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
    ani = ex.viz.animate_state_2d_facet(
        trj,
        vlim=vlim,
        titles=[f"{name} channel {i}" for i in range(num_channels)],
        grid=(1, num_channels),
        figsize=(5 * num_channels, 5),
    )

    ani.save(img_folder / f"{name}_2d.mp4")
    del ani

    p_meter_2d.update(1)

p_meter_2d.close()


p_meter_3d = tqdm(CONFIGURATIONS_3D, desc="", total=len(CONFIGURATIONS_3D))
# 3d problems (produce animations)
for stepper_3d, name, ic_distribution, warmup_steps, steps, vlim in CONFIGURATIONS_3D:
    p_meter_3d.set_description(f"3d {name}")

    ic = ic_distribution(stepper_3d.num_points, key=ic_key)
    ic = ex.repeat(stepper_3d, warmup_steps)(ic)
    trj = ex.rollout(stepper_3d, steps, include_init=True)(ic)
    jnp.save(img_folder / f"{name}_3d.npy", trj)

    if HAS_VAPE:
        num_channels = stepper_3d.num_channels
        ani = ex.viz.animate_state_3d_facet(
            trj,
            vlim=vlim,
            titles=[f"{name} channel {i}" for i in range(num_channels)],
            grid=(1, num_channels),
            figsize=(5 * num_channels, 5),
        )

        ani.save(img_folder / f"{name}_3d.mp4")
        del ani

    p_meter_3d.update(1)

p_meter_3d.close()
