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


# ===========================================================================
# Navier-Stokes vorticity tests
# ===========================================================================


class TestNavierStokesVorticity:
    def test_non_2d_raises(self):
        """NavierStokesVorticity only supports 2D."""
        with pytest.raises(ValueError, match="2"):
            ex.stepper.NavierStokesVorticity(1, 1.0, 32, 0.01)
        with pytest.raises(ValueError, match="2"):
            ex.stepper.NavierStokesVorticity(3, 1.0, 32, 0.01)

    def test_step_produces_finite_output(self):
        """A single step should produce finite (non-NaN, non-Inf) output."""
        stepper = ex.stepper.NavierStokesVorticity(2, 1.0, 32, 0.01, diffusivity=0.01)
        u_0 = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5)(
            32, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        assert u_1.shape == u_0.shape
        assert jnp.all(jnp.isfinite(u_1))

    def test_output_shape(self):
        """Output should have shape (1, N, N)."""
        N = 32
        stepper = ex.stepper.NavierStokesVorticity(2, 1.0, N, 0.01)
        u_0 = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5)(
            N, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        assert u_1.shape == (1, N, N)

    def test_diffusion_decays_energy(self):
        """With high diffusivity and no convection, energy should decay."""
        stepper = ex.stepper.NavierStokesVorticity(
            2,
            1.0,
            32,
            0.01,
            diffusivity=0.1,
            vorticity_convection_scale=0.0,  # pure diffusion
        )
        u_0 = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5)(
            32, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        # Energy = L2 norm squared
        energy_0 = float(jnp.sum(u_0**2))
        energy_1 = float(jnp.sum(u_1**2))
        assert energy_1 < energy_0

    def test_zero_diffusivity_preserves_more_energy(self):
        """With very low diffusivity, energy should be preserved better."""
        u_0 = ex.ic.RandomTruncatedFourierSeries(2, cutoff=3)(
            32, key=jax.random.PRNGKey(0)
        )
        stepper_high_visc = ex.stepper.NavierStokesVorticity(
            2, 1.0, 32, 0.001, diffusivity=0.1
        )
        stepper_low_visc = ex.stepper.NavierStokesVorticity(
            2, 1.0, 32, 0.001, diffusivity=0.001
        )
        u_high = stepper_high_visc(u_0)
        u_low = stepper_low_visc(u_0)
        # Low viscosity should preserve more energy
        energy_high = float(jnp.sum(u_high**2))
        energy_low = float(jnp.sum(u_low**2))
        assert energy_low > energy_high

    def test_drag_accelerates_decay(self):
        """Positive drag (λ>0 means amplification in the linear operator, but
        negative drag λ<0 means additional damping)."""
        u_0 = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5)(
            32, key=jax.random.PRNGKey(0)
        )
        stepper_no_drag = ex.stepper.NavierStokesVorticity(
            2,
            1.0,
            32,
            0.01,
            diffusivity=0.01,
            drag=0.0,
            vorticity_convection_scale=0.0,
        )
        stepper_drag = ex.stepper.NavierStokesVorticity(
            2,
            1.0,
            32,
            0.01,
            diffusivity=0.01,
            drag=-1.0,
            vorticity_convection_scale=0.0,
        )
        u_no_drag = stepper_no_drag(u_0)
        u_drag = stepper_drag(u_0)
        energy_no_drag = float(jnp.sum(u_no_drag**2))
        energy_drag = float(jnp.sum(u_drag**2))
        assert energy_drag < energy_no_drag


class TestKolmogorovFlowVorticity:
    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2"):
            ex.stepper.KolmogorovFlowVorticity(1, 1.0, 32, 0.01)

    def test_step_produces_finite_output(self):
        stepper = ex.stepper.KolmogorovFlowVorticity(
            2, 2 * jnp.pi, 64, 0.01, diffusivity=0.01
        )
        u_0 = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5)(
            64, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        assert u_1.shape == u_0.shape
        assert jnp.all(jnp.isfinite(u_1))

    def test_injection_adds_energy(self):
        """Kolmogorov forcing should inject energy (compared to unforced NS)."""
        N = 64
        L = 2 * jnp.pi
        dt = 0.01
        u_0 = 0.01 * ex.ic.RandomTruncatedFourierSeries(2, cutoff=3)(
            N, key=jax.random.PRNGKey(0)
        )

        # Unforced NS with same parameters
        ns_stepper = ex.stepper.NavierStokesVorticity(
            2,
            L,
            N,
            dt,
            diffusivity=0.01,
            drag=-0.1,
        )
        # Kolmogorov (forced)
        kolm_stepper = ex.stepper.KolmogorovFlowVorticity(
            2,
            L,
            N,
            dt,
            diffusivity=0.01,
            drag=-0.1,
            injection_mode=4,
            injection_scale=1.0,
        )

        # Run a few steps
        u_ns = u_0
        u_kolm = u_0
        for _ in range(10):
            u_ns = ns_stepper(u_ns)
            u_kolm = kolm_stepper(u_kolm)

        energy_ns = float(jnp.sum(u_ns**2))
        energy_kolm = float(jnp.sum(u_kolm**2))
        # The forced simulation should have more energy
        assert energy_kolm > energy_ns

    def test_multi_step_stable(self):
        """Multiple steps should remain stable (finite)."""
        stepper = ex.stepper.KolmogorovFlowVorticity(
            2, 2 * jnp.pi, 64, 0.01, diffusivity=0.01
        )
        u = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5)(
            64, key=jax.random.PRNGKey(0)
        )
        for _ in range(20):
            u = stepper(u)
        assert jnp.all(jnp.isfinite(u))


# # ===========================================================================
# # BelousovZhabotinsky tests (imported directly since not in public API)
# # ===========================================================================


# class TestBelousovZhabotinsky:
#     def test_instantiate(self):
#         """BZ stepper should instantiate without error."""
#         from exponax.stepper.reaction._belousov_zhabotinsky import (
#             BelousovZhabotinsky,
#         )

#         for D in [1, 2]:
#             BelousovZhabotinsky(D, 1.0, 32, 0.001)

#     def test_step_produces_finite_output(self):
#         """A single step should produce finite output."""
#         from exponax.stepper.reaction._belousov_zhabotinsky import (
#             BelousovZhabotinsky,
#         )

#         stepper = BelousovZhabotinsky(1, 1.0, 64, 0.001)
#         # BZ requires 3 channels and ICs in [0, 1]
#         key = jax.random.PRNGKey(0)
#         u_0 = jax.random.uniform(key, (3, 64), minval=0.0, maxval=0.5)
#         u_1 = stepper(u_0)
#         assert u_1.shape == (3, 64)
#         assert jnp.all(jnp.isfinite(u_1))

#     def test_requires_3_channels(self):
#         """BZ nonlinear fun should require exactly 3 channels."""
#         from exponax.stepper.reaction._belousov_zhabotinsky import (
#             BelousovZhabotinskyNonlinearFun,
#         )

#         nonlin = BelousovZhabotinskyNonlinearFun(
#             num_spatial_dims=1,
#             num_points=32,
#             dealiasing_fraction=0.5,
#         )
#         # Provide 2-channel input (wrong) - shape: (2, N//2+1) complex
#         bad_input = jnp.zeros((2, 17), dtype=jnp.complex64)
#         with pytest.raises(ValueError, match="3"):
#             nonlin(bad_input)

#     def test_multi_step_stability(self):
#         """Multiple steps with small dt should remain finite."""
#         from exponax.stepper.reaction._belousov_zhabotinsky import (
#             BelousovZhabotinsky,
#         )

#         stepper = BelousovZhabotinsky(1, 1.0, 64, 0.0005)
#         key = jax.random.PRNGKey(42)
#         u = jax.random.uniform(key, (3, 64), minval=0.1, maxval=0.4)
#         for _ in range(20):
#             u = stepper(u)
#         assert jnp.all(jnp.isfinite(u)), "BZ stepper produced NaN/Inf after 20 steps"

#     def test_2d_instantiate_and_step(self):
#         """BZ should also work in 2D."""
#         from exponax.stepper.reaction._belousov_zhabotinsky import (
#             BelousovZhabotinsky,
#         )

#         stepper = BelousovZhabotinsky(2, 1.0, 32, 0.001)
#         key = jax.random.PRNGKey(0)
#         u_0 = jax.random.uniform(key, (3, 32, 32), minval=0.0, maxval=0.5)
#         u_1 = stepper(u_0)
#         assert u_1.shape == (3, 32, 32)
#         assert jnp.all(jnp.isfinite(u_1))
