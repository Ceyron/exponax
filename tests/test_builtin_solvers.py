import jax
import jax.numpy as jnp
import pytest

import exponax as ex


def test_instantiate():
    domain_extent = 10.0
    num_points = 25
    dt = 0.1

    for num_spatial_dims in [1, 2, 3]:
        for simulator in [
            ex.stepper.Advection,
            ex.stepper.Diffusion,
            ex.stepper.AdvectionDiffusion,
            ex.stepper.Dispersion,
            ex.stepper.HyperDiffusion,
            ex.stepper.Burgers,
            ex.stepper.KuramotoSivashinsky,
            ex.stepper.KuramotoSivashinskyConservative,
            ex.stepper.KortewegDeVries,
            ex.stepper.generic.GeneralConvectionStepper,
            ex.stepper.generic.GeneralGradientNormStepper,
            ex.stepper.generic.GeneralLinearStepper,
            ex.stepper.generic.GeneralNonlinearStepper,
            ex.stepper.generic.GeneralPolynomialStepper,
        ]:
            simulator(num_spatial_dims, domain_extent, num_points, dt)

    for num_spatial_dims in [1, 2, 3]:
        for simulator in [
            ex.stepper.reaction.FisherKPP,
            ex.stepper.reaction.AllenCahn,
            ex.stepper.reaction.CahnHilliard,
            ex.stepper.reaction.SwiftHohenberg,
            # ex.stepper.reaction.BelousovZhabotinsky,
            ex.stepper.reaction.GrayScott,
        ]:
            simulator(num_spatial_dims, domain_extent, num_points, dt)

    for simulator in [
        ex.stepper.NavierStokesVorticity,
        ex.stepper.KolmogorovFlowVorticity,
    ]:
        simulator(2, domain_extent, num_points, dt)

    for num_spatial_dims in [1, 2, 3]:
        ex.poisson.Poisson(num_spatial_dims, domain_extent, num_points)

    for num_spatial_dims in [1, 2, 3]:
        for normalized_simulator in [
            ex.stepper.generic.NormalizedLinearStepper,
            ex.stepper.generic.NormalizedConvectionStepper,
            ex.stepper.generic.NormalizedGradientNormStepper,
            ex.stepper.generic.NormalizedPolynomialStepper,
            ex.stepper.generic.NormalizedNonlinearStepper,
        ]:
            normalized_simulator(num_spatial_dims, num_points)


@pytest.mark.parametrize(
    "specific_stepper,general_stepper_coefficients",
    [
        # Linear problems
        (
            ex.stepper.Advection(1, 3.0, 50, 0.1, velocity=1.0),
            [0.0, -1.0],
        ),
        (
            ex.stepper.Diffusion(1, 3.0, 50, 0.1, diffusivity=0.01),
            [0.0, 0.0, 0.01],
        ),
        (
            ex.stepper.AdvectionDiffusion(
                1, 3.0, 50, 0.1, velocity=1.0, diffusivity=0.01
            ),
            [0.0, -1.0, 0.01],
        ),
        (
            ex.stepper.Dispersion(1, 3.0, 50, 0.1, dispersivity=0.0001),
            [0.0, 0.0, 0.0, 0.0001],
        ),
        (
            ex.stepper.HyperDiffusion(1, 3.0, 50, 0.1, hyper_diffusivity=0.00001),
            [0.0, 0.0, 0.0, 0.0, -0.00001],
        ),
    ],
)
def test_specific_stepper_to_general_linear_stepper(
    specific_stepper,
    general_stepper_coefficients,
):
    num_spatial_dims = specific_stepper.num_spatial_dims
    domain_extent = specific_stepper.domain_extent
    num_points = specific_stepper.num_points
    dt = specific_stepper.dt

    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    general_stepper = ex.stepper.generic.GeneralLinearStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        linear_coefficients=general_stepper_coefficients,
    )

    specific_pred = specific_stepper(u_0)
    general_pred = general_stepper(u_0)

    assert specific_pred == pytest.approx(general_pred, rel=1e-4)


@pytest.mark.parametrize(
    "specific_stepper,general_stepper_scale,general_stepper_coefficients,conservative",
    [
        # Linear problems
        (
            ex.stepper.Advection(1, 3.0, 50, 0.1, velocity=1.0),
            0.0,
            [0.0, -1.0],
            False,
        ),
        (
            ex.stepper.Diffusion(1, 3.0, 50, 0.1, diffusivity=0.01),
            0.0,
            [0.0, 0.0, 0.01],
            False,
        ),
        (
            ex.stepper.AdvectionDiffusion(
                1, 3.0, 50, 0.1, velocity=1.0, diffusivity=0.01
            ),
            0.0,
            [0.0, -1.0, 0.01],
            False,
        ),
        (
            ex.stepper.Dispersion(1, 3.0, 50, 0.1, dispersivity=0.0001),
            0.0,
            [0.0, 0.0, 0.0, 0.0001],
            False,
        ),
        (
            ex.stepper.HyperDiffusion(1, 3.0, 50, 0.1, hyper_diffusivity=0.00001),
            0.0,
            [0.0, 0.0, 0.0, 0.0, -0.00001],
            False,
        ),
        # nonlinear problems
        (
            ex.stepper.Burgers(1, 3.0, 50, 0.1, diffusivity=0.05, convection_scale=1.0),
            1.0,
            [0.0, 0.0, 0.05],
            False,
        ),
        (
            ex.stepper.KortewegDeVries(
                1, 3.0, 50, 0.1, dispersivity=1.0, convection_scale=-6.0
            ),
            -6.0,
            [0.0, 0.0, 0.0, -1.0, -0.01],
            False,
        ),
        (
            ex.stepper.KuramotoSivashinskyConservative(
                1,
                3.0,
                50,
                0.1,
                convection_scale=1.0,
                second_order_scale=1.0,
                fourth_order_scale=1.0,
            ),
            1.0,
            [0.0, 0.0, -1.0, 0.0, -1.0],
            True,
        ),
    ],
)
def test_specific_stepper_to_general_convection_stepper(
    specific_stepper,
    general_stepper_scale,
    general_stepper_coefficients,
    conservative,
):
    num_spatial_dims = specific_stepper.num_spatial_dims
    domain_extent = specific_stepper.domain_extent
    num_points = specific_stepper.num_points
    dt = specific_stepper.dt

    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    general_stepper = ex.stepper.generic.GeneralConvectionStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        linear_coefficients=general_stepper_coefficients,
        convection_scale=general_stepper_scale,
        conservative=conservative,
    )

    specific_pred = specific_stepper(u_0)
    general_pred = general_stepper(u_0)

    assert specific_pred == pytest.approx(general_pred, rel=1e-4)


@pytest.mark.parametrize(
    "specific_stepper,general_stepper_scale,general_stepper_coefficients",
    [
        # Linear problems
        (
            ex.stepper.Advection(1, 3.0, 50, 0.1, velocity=1.0),
            0.0,
            [0.0, -1.0],
        ),
        (
            ex.stepper.Diffusion(1, 3.0, 50, 0.1, diffusivity=0.01),
            0.0,
            [0.0, 0.0, 0.01],
        ),
        (
            ex.stepper.AdvectionDiffusion(
                1, 3.0, 50, 0.1, velocity=1.0, diffusivity=0.01
            ),
            0.0,
            [0.0, -1.0, 0.01],
        ),
        (
            ex.stepper.Dispersion(1, 3.0, 50, 0.1, dispersivity=0.0001),
            0.0,
            [0.0, 0.0, 0.0, 0.0001],
        ),
        (
            ex.stepper.HyperDiffusion(1, 3.0, 50, 0.1, hyper_diffusivity=0.00001),
            0.0,
            [0.0, 0.0, 0.0, 0.0, -0.00001],
        ),
        # nonlinear problems
        (
            ex.stepper.KuramotoSivashinsky(
                1,
                3.0,
                50,
                0.1,
                gradient_norm_scale=1.0,
                second_order_scale=1.0,
                fourth_order_scale=1.0,
            ),
            1.0,
            [0.0, 0.0, -1.0, 0.0, -1.0],
        ),
    ],
)
def test_specific_to_general_gradient_norm_stepper(
    specific_stepper,
    general_stepper_scale,
    general_stepper_coefficients,
):
    num_spatial_dims = specific_stepper.num_spatial_dims
    domain_extent = specific_stepper.domain_extent
    num_points = specific_stepper.num_points
    dt = specific_stepper.dt

    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    general_stepper = ex.stepper.generic.GeneralGradientNormStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        linear_coefficients=general_stepper_coefficients,
        gradient_norm_scale=general_stepper_scale,
    )

    specific_pred = specific_stepper(u_0)
    general_pred = general_stepper(u_0)

    assert specific_pred == pytest.approx(general_pred, rel=1e-4)


@pytest.mark.parametrize(
    "coefficients",
    [
        [
            0.5,
        ],  # drag
        [0.0, -0.3],  # advection
        [0.0, 0.0, 0.01],  # diffusion
        [0.0, -0.2, 0.01],  # advection-diffusion
        [0.0, 0.0, 0.0, 0.001],  # dispersion
        [0.0, 0.0, 0.0, 0.0, -0.0001],  # hyperdiffusion
    ],
)
def test_linear_normalized_stepper(coefficients):
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 50
    dt = 0.1

    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    regular_linear_stepper = ex.stepper.generic.GeneralLinearStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        linear_coefficients=coefficients,
    )
    normalized_linear_stepper = ex.stepper.generic.NormalizedLinearStepper(
        num_spatial_dims,
        num_points,
        normalized_linear_coefficients=ex.stepper.generic.normalize_coefficients(
            coefficients,
            domain_extent=domain_extent,
            dt=dt,
        ),
    )

    regular_linear_pred = regular_linear_stepper(u_0)
    normalized_linear_pred = normalized_linear_stepper(u_0)

    assert regular_linear_pred == pytest.approx(normalized_linear_pred, rel=1e-4)


def test_nonlinear_normalized_stepper():
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 50
    dt = 0.1
    diffusivity = 0.1
    convection_scale = 1.0

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent) + 0.3

    regular_burgers_stepper = ex.stepper.Burgers(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        diffusivity=diffusivity,
        convection_scale=convection_scale,
    )
    normalized_burgers_stepper = ex.stepper.generic.NormalizedConvectionStepper(
        num_spatial_dims,
        num_points,
        normalized_linear_coefficients=ex.stepper.generic.normalize_coefficients(
            [0.0, 0.0, diffusivity],
            domain_extent=domain_extent,
            dt=dt,
        ),
        normalized_convection_scale=ex.stepper.generic.normalize_convection_scale(
            convection_scale,
            domain_extent=domain_extent,
            dt=dt,
        ),
    )

    regular_burgers_pred = regular_burgers_stepper(u_0)
    normalized_burgers_pred = normalized_burgers_stepper(u_0)

    assert regular_burgers_pred == pytest.approx(
        normalized_burgers_pred, rel=1e-5, abs=1e-5
    )
