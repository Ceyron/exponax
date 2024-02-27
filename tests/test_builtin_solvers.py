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
