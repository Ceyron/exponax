import jax
import pytest

import exponax as ex


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_kdv(num_spatial_dims: int):
    DOMAIN_EXTENT = 5.0
    NUM_POINTS = 48
    DT = 0.01
    DIFFUSIVITY = 0.1
    DISPERSIVITY = 0.001
    HYPER_DIFFUSIVITY = 0.0001

    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=num_spatial_dims, max_one=True
    )(NUM_POINTS, key=jax.random.PRNGKey(0))

    kdv_stepper_only_viscous = ex.stepper.KortewegDeVries(
        num_spatial_dims=num_spatial_dims,
        domain_extent=DOMAIN_EXTENT,
        num_points=NUM_POINTS,
        dt=DT,
        convection_scale=0.0,
        diffusivity=DIFFUSIVITY,
        dispersivity=0.0,
        hyper_diffusivity=0.0,
        advect_over_diffuse=False,
        diffuse_over_diffuse=False,
        single_channel=True,
    )
    diffusion_stepper = ex.stepper.Diffusion(
        num_spatial_dims=num_spatial_dims,
        domain_extent=DOMAIN_EXTENT,
        num_points=NUM_POINTS,
        dt=DT,
        diffusivity=DIFFUSIVITY,
    )

    assert kdv_stepper_only_viscous(u_0) == pytest.approx(
        diffusion_stepper(u_0), abs=1e-6
    )

    kdv_stepper_only_dispersion = ex.stepper.KortewegDeVries(
        num_spatial_dims=num_spatial_dims,
        domain_extent=DOMAIN_EXTENT,
        num_points=NUM_POINTS,
        dt=DT,
        convection_scale=0.0,
        diffusivity=0.0,
        dispersivity=-DISPERSIVITY,
        hyper_diffusivity=0.0,
        advect_over_diffuse=False,
        diffuse_over_diffuse=False,
        single_channel=True,
    )
    dispersion_stepper = ex.stepper.Dispersion(
        num_spatial_dims=num_spatial_dims,
        domain_extent=DOMAIN_EXTENT,
        num_points=NUM_POINTS,
        dt=DT,
        dispersivity=DISPERSIVITY,
    )

    assert kdv_stepper_only_dispersion(u_0) == pytest.approx(
        dispersion_stepper(u_0), abs=1e-6
    )

    kdv_stepper_only_hyper_diffusion = ex.stepper.KortewegDeVries(
        num_spatial_dims=num_spatial_dims,
        domain_extent=DOMAIN_EXTENT,
        num_points=NUM_POINTS,
        dt=DT,
        convection_scale=0.0,
        diffusivity=0.0,
        dispersivity=0.0,
        hyper_diffusivity=HYPER_DIFFUSIVITY,
        advect_over_diffuse=False,
        diffuse_over_diffuse=False,
        single_channel=True,
    )
    hyper_diffusion_stepper = ex.stepper.HyperDiffusion(
        num_spatial_dims=num_spatial_dims,
        domain_extent=DOMAIN_EXTENT,
        num_points=NUM_POINTS,
        dt=DT,
        hyper_diffusivity=HYPER_DIFFUSIVITY,
    )

    assert kdv_stepper_only_hyper_diffusion(u_0) == pytest.approx(
        hyper_diffusion_stepper(u_0), abs=1e-6
    )
