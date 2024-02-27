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
            ex.Advection,
            ex.Diffusion,
            ex.AdvectionDiffusion,
            ex.Dispersion,
            ex.HyperDiffusion,
            ex.Burgers,
            ex.KuramotoSivashinsky,
            ex.KuramotoSivashinskyConservative,
            ex.SwiftHohenberg,
            ex.GrayScott,
            ex.KortevegDeVries,
            ex.FisherKPP,
            ex.AllenCahn,
            ex.CahnHilliard,
        ]:
            simulator(num_spatial_dims, domain_extent, num_points, dt)

    for simulator in [
        ex.NavierStokesVorticity2d,
        ex.KolmogorovFlowVorticity2d,
    ]:
        simulator(domain_extent, num_points, dt)

    for num_spatial_dims in [1, 2, 3]:
        ex.Poisson(num_spatial_dims, domain_extent, num_points)

    for num_spatial_dims in [1, 2, 3]:
        for normalized_simulator in [
            ex.NormalizedLinearStepper,
            ex.NormalizedConvectionStepper,
            ex.NormalizedGradientNormStepper,
        ]:
            normalized_simulator(num_spatial_dims, num_points)

@pytest.mark.parametrize(
    "specific_stepper,general_stepper_coefficients",
    [
        # Linear problems
        (
            ex.Advection(1, 3.0, 50, 0.1, velocity=1.0),
            [0.0, -1.0],
        ),
        (
            ex.Diffusion(1, 3.0, 50, 0.1, diffusivity=0.01),
            [0.0, 0.0, 0.01],
        ),
        (
            ex.AdvectionDiffusion(1, 3.0, 50, 0.1, velocity=1.0, diffusivity=0.01),
            [0.0, -1.0, 0.01],
        ),
        (
            ex.Dispersion(1, 3.0, 50, 0.1, dispersivity=0.0001),
            [0.0, 0.0, 0.0, 0.0001],
        ),
        (
            ex.HyperDiffusion(1, 3.0, 50, 0.1, hyper_diffusivity=0.00001),
            [0.0, 0.0, 0.0, 0.0, -0.00001],
        ),
    ]
)
def test_specific_stepper_to_general_linear_stepper(
    specific_stepper,
    general_stepper_coefficients,
):
    num_spatial_dims = specific_stepper.num_spatial_dims
    domain_extent = specific_stepper.domain_extent
    num_points = specific_stepper.num_points
    dt = specific_stepper.dt

    u_0 = ex.RandomTruncatedFourierSeries(
        num_spatial_dims,
        domain_extent,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    general_stepper = ex.GeneralLinearStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        coefficients=general_stepper_coefficients,
    )

    specific_pred = specific_stepper(u_0)
    general_pred = general_stepper(u_0)

    assert specific_pred == pytest.approx(general_pred, rel=1e-4)

@pytest.mark.parametrize(
    "specific_stepper,general_stepper_scale,general_stepper_coefficients",
    [
        # Linear problems
        (
            ex.Advection(1, 3.0, 50, 0.1, velocity=1.0),
            0.0,
            [0.0, -1.0],
        ),
        (
            ex.Diffusion(1, 3.0, 50, 0.1, diffusivity=0.01),
            0.0,
            [0.0, 0.0, 0.01],
        ),
        (
            ex.AdvectionDiffusion(1, 3.0, 50, 0.1, velocity=1.0, diffusivity=0.01),
            0.0,
            [0.0, -1.0, 0.01],
        ),
        (
            ex.Dispersion(1, 3.0, 50, 0.1, dispersivity=0.0001),
            0.0,
            [0.0, 0.0, 0.0, 0.0001],
        ),
        (
            ex.HyperDiffusion(1, 3.0, 50, 0.1, hyper_diffusivity=0.00001),
            0.0,
            [0.0, 0.0, 0.0, 0.0, -0.00001],
        ),
        # nonlinear problems
        (
            ex.Burgers(1, 3.0, 50, 0.1, diffusivity=0.05, convection_scale=1.0),
            1.0,
            [0.0, 0.0, 0.05],
        ),
        (
            ex.KortevegDeVries(1, 3.0, 50, 0.1, pure_dispersivity=1.0, convection_scale=-6.0),
            -6.0,
            [0.0, 0.0, 0.0, -1.0]
        ),
        (
            ex.KuramotoSivashinskyConservative(1, 3.0, 50, 0.1, convection_scale=1.0, second_order_diffusivity=1.0, fourth_order_diffusivity=1.0),
            1.0,
            [0.0, 0.0, -1.0, 0.0, -1.0]
        )
    ]
)
def test_specific_stepper_to_general_convection_stepper(
    specific_stepper,
    general_stepper_scale,
    general_stepper_coefficients,
):
    num_spatial_dims = specific_stepper.num_spatial_dims
    domain_extent = specific_stepper.domain_extent
    num_points = specific_stepper.num_points
    dt = specific_stepper.dt

    u_0 = ex.RandomTruncatedFourierSeries(
        num_spatial_dims,
        domain_extent,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    general_stepper = ex.GeneralConvectionStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        coefficients=general_stepper_coefficients,
        convection_scale=general_stepper_scale,
    )

    specific_pred = specific_stepper(u_0)
    general_pred = general_stepper(u_0)

    assert specific_pred == pytest.approx(general_pred, rel=1e-4)

@pytest.mark.parametrize(
    "specific_stepper,general_stepper_scale,general_stepper_coefficients",
    [
        # Linear problems
        (
            ex.Advection(1, 3.0, 50, 0.1, velocity=1.0),
            0.0,
            [0.0, -1.0],
        ),
        (
            ex.Diffusion(1, 3.0, 50, 0.1, diffusivity=0.01),
            0.0,
            [0.0, 0.0, 0.01],
        ),
        (
            ex.AdvectionDiffusion(1, 3.0, 50, 0.1, velocity=1.0, diffusivity=0.01),
            0.0,
            [0.0, -1.0, 0.01],
        ),
        (
            ex.Dispersion(1, 3.0, 50, 0.1, dispersivity=0.0001),
            0.0,
            [0.0, 0.0, 0.0, 0.0001],
        ),
        (
            ex.HyperDiffusion(1, 3.0, 50, 0.1, hyper_diffusivity=0.00001),
            0.0,
            [0.0, 0.0, 0.0, 0.0, -0.00001],
        ),
        # nonlinear problems
        (
            ex.KuramotoSivashinsky(1, 3.0, 50, 0.1, gradient_norm_scale=1.0, second_order_diffusivity=1.0, fourth_order_diffusivity=1.0),
            1.0,
            [0.0, 0.0, -1.0, 0.0, -1.0]
        )
    ]
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

    u_0 = ex.RandomTruncatedFourierSeries(
        num_spatial_dims,
        domain_extent,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    general_stepper = ex.GeneralGradientNormStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        coefficients=general_stepper_coefficients,
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
    ]
)
def test_linear_normalized_stepper(coefficients):
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 50
    dt = 0.1

    u_0 = ex.RandomTruncatedFourierSeries(
        num_spatial_dims,
        domain_extent,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    regular_linear_stepper = ex.GeneralLinearStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        coefficients=coefficients,
    )
    normalized_linear_stepper = ex.NormalizedLinearStepper(
        num_spatial_dims,
        num_points,
        normalized_coefficients=ex.normalize_coefficients(
            domain_extent,
            coefficients,
        ),
        dt=dt,
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

    grid = ex.get_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent) + 0.3

    regular_burgers_stepper = ex.Burgers(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        diffusivity=diffusivity,
        convection_scale=convection_scale,
    )
    normalized_burgers_stepper = ex.NormalizedConvectionStepper(
        num_spatial_dims,
        num_points,
        dt=dt,
        normalized_coefficients=ex.normalize_coefficients(
            domain_extent,
            [0.0, 0.0, diffusivity],
        ),
        normalized_convection_scale=ex.normalize_convection_scale(
            domain_extent,
            convection_scale,
        ),
    )

    regular_burgers_pred = regular_burgers_stepper(u_0)
    normalized_burgers_pred = normalized_burgers_stepper(u_0)

    assert regular_burgers_pred == pytest.approx(normalized_burgers_pred)
