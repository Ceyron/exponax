import jax
import jax.numpy as jnp
import pytest

import exponax as ex


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_constant_offset(num_spatial_dims: int):
    DOMAIN_EXTENT = 5.0
    NUM_POINTS = 40
    grid = ex.make_grid(num_spatial_dims, DOMAIN_EXTENT, NUM_POINTS)

    u_0 = 2.0 * jnp.ones_like(grid[0:1])
    u_1 = 4.0 * jnp.ones_like(grid[0:1])

    assert ex.metrics.MSE(u_1, u_0, domain_extent=1.0) == pytest.approx(4.0)
    assert ex.metrics.MSE(u_1, u_0, domain_extent=DOMAIN_EXTENT) == pytest.approx(
        DOMAIN_EXTENT**num_spatial_dims * 4.0
    )

    # MSE metric is symmetric
    assert ex.metrics.MSE(u_0, u_1, domain_extent=1.0) == ex.metrics.MSE(
        u_1, u_0, domain_extent=1.0
    )
    assert ex.metrics.MSE(u_0, u_1, domain_extent=DOMAIN_EXTENT) == ex.metrics.MSE(
        u_1, u_0, domain_extent=DOMAIN_EXTENT
    )

    # == approx(1.0)
    assert ex.metrics.nMSE(u_1, u_0) == pytest.approx((4.0 - 2.0) ** 2 / (2.0) ** 2)
    assert ex.metrics.nMSE(u_1, u_0) == pytest.approx(1.0)

    # == approx (1/4)
    assert ex.metrics.nMSE(u_0, u_1) == pytest.approx((2.0 - 4.0) ** 2 / (4.0) ** 2)
    assert ex.metrics.nMSE(u_0, u_1) == pytest.approx(1 / 4)

    # == approx(0.4
    assert ex.metrics.sMSE(u_1, u_0) == pytest.approx(
        2.0 * (4.0 - 2.0) ** 2 / ((2.0) ** 2 + (4.0) ** 2)
    )
    assert ex.metrics.sMSE(u_1, u_0) == pytest.approx(0.4)

    # Symmetric metric must be symmetric
    assert ex.metrics.sMSE(u_0, u_1) == ex.metrics.sMSE(u_1, u_0)

    assert ex.metrics.RMSE(u_1, u_0, domain_extent=1.0) == pytest.approx(2.0)
    assert ex.metrics.RMSE(u_1, u_0, domain_extent=DOMAIN_EXTENT) == pytest.approx(
        jnp.sqrt(DOMAIN_EXTENT**num_spatial_dims * 4.0)
    )

    # RMSE is symmetric
    assert ex.metrics.RMSE(u_0, u_1, domain_extent=1.0) == ex.metrics.RMSE(
        u_1, u_0, domain_extent=1.0
    )
    assert ex.metrics.RMSE(u_0, u_1, domain_extent=DOMAIN_EXTENT) == ex.metrics.RMSE(
        u_1, u_0, domain_extent=DOMAIN_EXTENT
    )

    # == approx(1.0)
    assert ex.metrics.nRMSE(u_1, u_0) == pytest.approx(
        jnp.sqrt((4.0 - 2.0) ** 2 / 2.0**2)
    )
    assert ex.metrics.nRMSE(u_1, u_0) == pytest.approx(1.0)

    # == approx(sqrt(1/4)) == approx(0.5)
    assert ex.metrics.nRMSE(u_0, u_1) == pytest.approx(
        jnp.sqrt((2.0 - 4.0) ** 2 / 4.0**2)
    )
    assert ex.metrics.nRMSE(u_0, u_1) == pytest.approx(0.5)

    # == approx(2/3)
    assert ex.metrics.sRMSE(u_1, u_0) == pytest.approx(2 / 3)

    # The Fourier nRMSE should be identical to the spatial nRMSE
    # assert ex.metrics.fourier_nRMSE(u_1, u_0) == ex.metrics.nRMSE(u_1, u_0)
    # assert ex.metrics.fourier_nRMSE(u_0, u_1) == ex.metrics.nRMSE(u_0, u_1)

    # The Fourier based losses must be similar to their spatial counterparts due
    # to Parseval's identity
    assert ex.metrics.fourier_MSE(u_1, u_0) == pytest.approx(ex.metrics.MSE(u_1, u_0))
    assert ex.metrics.fourier_MSE(u_0, u_1) == pytest.approx(ex.metrics.MSE(u_0, u_1))
    # This equivalence does not hold for the MAE
    # assert ex.metrics.fourier_MAE(u_1, u_0) == pytest.approx(ex.metrics.MAE(u_1, u_0))
    # assert ex.metrics.fourier_MAE(u_0, u_1) == pytest.approx(ex.metrics.MAE(u_0, u_1))
    assert ex.metrics.fourier_RMSE(u_1, u_0) == pytest.approx(ex.metrics.RMSE(u_1, u_0))
    assert ex.metrics.fourier_RMSE(u_0, u_1) == pytest.approx(ex.metrics.RMSE(u_0, u_1))


def test_fourier_losses():
    # Test specific features of Fourier-based losses like filtering and
    # derivatives
    pass


@pytest.mark.parametrize(
    "num_spatial_dims,ic_gen",
    [
        (num_spatial_dims, ic_gen)
        for num_spatial_dims in [1, 2, 3]
        for ic_gen in [
            ex.ic.RandomTruncatedFourierSeries(num_spatial_dims, offset_range=(-1, 1)),
        ]
    ],
)
def test_fourier_equals_spatial_aggregation(num_spatial_dims, ic_gen):
    """
    Must be identical due to Parseval's identity
    """
    NUM_POINTS = 40
    DOMAIN_EXTENT = 5.0

    u_0 = ic_gen(NUM_POINTS, key=jax.random.PRNGKey(0))
    u_1 = ic_gen(NUM_POINTS, key=jax.random.PRNGKey(1))

    assert ex.metrics.fourier_MSE(
        u_1, u_0, domain_extent=DOMAIN_EXTENT
    ) == pytest.approx(ex.metrics.MSE(u_1, u_0, domain_extent=DOMAIN_EXTENT))
    # # This equivalence does not hold for the MAE
    # assert (
    #   ex.metrics.fourier_MAE(u_1, u_0, domain_extent=DOMAIN_EXTENT)
    #   == pytest.approx(
    #     ex.metrics.MAE(u_1, u_0, domain_extent=DOMAIN_EXTENT)
    # )
    assert ex.metrics.fourier_RMSE(
        u_1, u_0, domain_extent=DOMAIN_EXTENT
    ) == pytest.approx(ex.metrics.RMSE(u_1, u_0, domain_extent=DOMAIN_EXTENT))


@pytest.mark.parametrize(
    "num_spatial_dims,num_points",
    [
        (num_spatial_dims, num_points)
        for num_spatial_dims in [1, 2, 3]
        for num_points in [40, 41]
    ],
)
def test_fourier_metric_filtering(num_spatial_dims, num_points):
    # It is sufficient only test one fourier_XXXXX metric because they all use
    # exponax.metrics.fourier_aggegator to perform the filtering
    DOMAIN_EXTENT = 2 * jnp.pi
    grid = ex.make_grid(num_spatial_dims, DOMAIN_EXTENT, num_points)

    u = jnp.sin(4 * grid[0:1])
    if num_spatial_dims > 1:
        u *= jnp.sin(4 * grid[1:2])
    if num_spatial_dims > 2:
        u *= jnp.sin(4 * grid[2:3])

    # If all modi are included, metric must be non-zero
    assert float(ex.metrics.fourier_MSE(u)) != pytest.approx(0.0, abs=1e-6)
    # If the lower bound is higher than the active modi, the metric must be zero
    assert float(ex.metrics.fourier_MSE(u, low=8)) == pytest.approx(0.0, abs=1e-6)
    # If the upper bound is higher than the active modi, the metric must be non-zero
    assert float(ex.metrics.fourier_MSE(u, high=8)) != pytest.approx(0.0, abs=1e-6)
    # If the upper bound is lower than the active modi, the metric must be zero
    assert float(ex.metrics.fourier_MSE(u, high=2)) == pytest.approx(0.0, abs=1e-6)
    # If the lower bound is lower than the active modi, the metric must be non-zero
    assert float(ex.metrics.fourier_MSE(u, low=2)) != pytest.approx(0.0, abs=1e-6)
    # If the selected frequency interval includes all active modi, the metric
    # must be non-zero
    assert float(ex.metrics.fourier_MSE(u, low=2, high=8)) != pytest.approx(
        0.0, abs=1e-6
    )
    # If the selected frequency interval only considers inactive modi, the metric
    # must be zero
    assert float(ex.metrics.fourier_MSE(u, low=8, high=16)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(ex.metrics.fourier_MSE(u, low=0, high=2)) == pytest.approx(
        0.0, abs=1e-6
    )
    # If the active modi is on the lower end of frequency space, the metric must
    # be non-zero
    assert float(ex.metrics.fourier_MSE(u, low=4, high=8)) != pytest.approx(
        0.0, abs=1e-6
    )
    # If the active modi is on the upper end of frequency space, the metric must
    # be non-zero
    assert float(ex.metrics.fourier_MSE(u, low=0, high=4)) != pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize(
    "num_spatial_dims,metric_fn_name",
    [
        (num_spatial_dims, metric_fn_name)
        for num_spatial_dims in [1, 2, 3]
        for metric_fn_name in [
            "MAE",
            "nMAE",
            "MSE",
            "nMSE",
            "RMSE",
            "nRMSE",
        ]
    ],
)
def test_sobolev_vs_manual(num_spatial_dims, metric_fn_name):
    NUM_POINTS = 40
    DOMAIN_EXTENT = 5.0

    ic_gen = ex.ic.RandomTruncatedFourierSeries(num_spatial_dims, offset_range=(-1, 1))
    u_0 = ic_gen(NUM_POINTS, key=jax.random.PRNGKey(0))
    u_1 = ic_gen(NUM_POINTS, key=jax.random.PRNGKey(1))

    fourier_metric_fn = getattr(ex.metrics, "fourier_" + metric_fn_name)
    sobolev_metric_fn = getattr(ex.metrics, "H1_" + metric_fn_name)

    correct_metric_value = fourier_metric_fn(
        u_0, u_1, domain_extent=DOMAIN_EXTENT
    ) + fourier_metric_fn(u_0, u_1, domain_extent=DOMAIN_EXTENT, derivative_order=1)

    assert sobolev_metric_fn(u_0, u_1, domain_extent=DOMAIN_EXTENT) == pytest.approx(
        correct_metric_value
    )


# # Below always evaluates to 2 * pi no matter the values of k and l
# def analytical_L2_diff_norm(k: int, l:int):
#     term1 = 2 * jnp.pi
#     term2 = -jnp.sin(4 * k * jnp.pi) / (4 * k)
#     term3 = (2 * (-(
#         l * jnp.cos(2 * l * jnp.pi) * jnp.sin(2 * k * jnp.pi)
#         ) + k * jnp.cos(2 * k * jnp.pi) * jnp.sin(2 * l * jnp.pi))
#     ) / (k**2 - l**2)
#     term4 = -jnp.sin(4 * l * jnp.pi) / (4 * l)

#     result = term1 + term2 + term3 + term4
#     return result


@pytest.mark.parametrize("wavenumber_k,wavenumber_l", [(1, 2), (2, 1), (3, 4)])
def test_analytical_solution_1d(wavenumber_k: int, wavenumber_l: int):
    NUM_POINTS = 100
    DOMAIN_EXTENT = 2 * jnp.pi

    grid = ex.make_grid(1, DOMAIN_EXTENT, NUM_POINTS)

    u_0 = jnp.sin(wavenumber_k * grid)
    u_1 = jnp.sin(wavenumber_l * grid)

    # assert ex.metrics.MSE(u_1, u_0, domain_extent=DOMAIN_EXTENT) == pytest.approx(
    #     analytical_L2_diff_norm(k, l)
    # )
    assert ex.metrics.MSE(u_1, u_0, domain_extent=DOMAIN_EXTENT) == pytest.approx(
        2 * jnp.pi
    )
    assert ex.metrics.nMSE(u_1, u_0) == pytest.approx(2.0)
    assert ex.metrics.nMSE(u_1, u_0, domain_extent=DOMAIN_EXTENT) == pytest.approx(2.0)


# ===========================================================================
# Correlation tests
# ===========================================================================


class TestCorrelation:
    def test_identical_fields(self):
        """Correlation of a field with itself should be 1.0."""
        u = jax.random.normal(jax.random.PRNGKey(0), (1, 64))
        assert float(ex.metrics.correlation(u, u)) == pytest.approx(1.0, abs=1e-5)

    def test_negative_field(self):
        """Correlation of a field with its negation should be -1.0."""
        u = jax.random.normal(jax.random.PRNGKey(0), (1, 64))
        assert float(ex.metrics.correlation(u, -u)) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_fields_1d(self):
        """Correlation of orthogonal sine waves should be ~0."""
        grid = ex.make_grid(1, 2 * jnp.pi, 128)
        u = jnp.sin(grid)
        v = jnp.cos(grid)
        assert float(ex.metrics.correlation(u, v)) == pytest.approx(0.0, abs=1e-4)

    def test_multi_channel(self):
        """Correlation should average over channels."""
        u = jax.random.normal(jax.random.PRNGKey(0), (3, 64))
        corr = float(ex.metrics.correlation(u, u))
        assert corr == pytest.approx(1.0, abs=1e-5)

    def test_2d(self):
        """Correlation should work in 2D."""
        u = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32))
        assert float(ex.metrics.correlation(u, u)) == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# mean_metric tests
# ===========================================================================


class TestMeanMetric:
    def test_basic(self):
        """mean_metric should average the metric over the batch axis."""
        u_pred = jax.random.normal(jax.random.PRNGKey(0), (5, 1, 64))
        u_ref = jax.random.normal(jax.random.PRNGKey(1), (5, 1, 64))
        result = ex.metrics.mean_metric(ex.metrics.MSE, u_pred, u_ref)
        # Manually compute
        manual = jnp.mean(jax.vmap(ex.metrics.MSE)(u_pred, u_ref))
        assert float(result) == pytest.approx(float(manual), abs=1e-6)

    def test_with_kwargs(self):
        """mean_metric should pass kwargs through."""
        u_pred = jax.random.normal(jax.random.PRNGKey(0), (5, 1, 64))
        u_ref = jax.random.normal(jax.random.PRNGKey(1), (5, 1, 64))
        result = ex.metrics.mean_metric(
            ex.metrics.MSE, u_pred, u_ref, domain_extent=2.0
        )
        manual = jnp.mean(
            jax.vmap(lambda p, r: ex.metrics.MSE(p, r, domain_extent=2.0))(
                u_pred, u_ref
            )
        )
        assert float(result) == pytest.approx(float(manual), abs=1e-6)

    def test_scalar_output(self):
        """mean_metric result should be a scalar."""
        u_pred = jax.random.normal(jax.random.PRNGKey(0), (3, 1, 32))
        u_ref = jax.random.normal(jax.random.PRNGKey(1), (3, 1, 32))
        result = ex.metrics.mean_metric(ex.metrics.RMSE, u_pred, u_ref)
        assert result.ndim == 0


# ===========================================================================
# MAE / nMAE / sMAE tests
# ===========================================================================


class TestMAEFamily:
    def test_mae_constant_fields(self):
        """MAE of constant fields with difference=2 on unit domain."""
        u_pred = 4.0 * jnp.ones((1, 64))
        u_ref = 2.0 * jnp.ones((1, 64))
        # MAE = sum (L/N)^D |u-v| = 1/64 * 64 * 2 = 2.0
        assert float(ex.metrics.MAE(u_pred, u_ref)) == pytest.approx(2.0, abs=1e-5)

    def test_mae_without_ref(self):
        """MAE without reference = norm of the state."""
        u = 3.0 * jnp.ones((1, 64))
        assert float(ex.metrics.MAE(u)) == pytest.approx(3.0, abs=1e-5)

    def test_nmae_constant_fields(self):
        """nMAE = |u-v|/|v| = 2/2 = 1.0 for constant fields."""
        u_pred = 4.0 * jnp.ones((1, 64))
        u_ref = 2.0 * jnp.ones((1, 64))
        assert float(ex.metrics.nMAE(u_pred, u_ref)) == pytest.approx(1.0, abs=1e-5)

    def test_smae_constant_fields(self):
        """sMAE = 2|u-v|/(|u|+|v|) = 2*2/(4+2) = 2/3."""
        u_pred = 4.0 * jnp.ones((1, 64))
        u_ref = 2.0 * jnp.ones((1, 64))
        assert float(ex.metrics.sMAE(u_pred, u_ref)) == pytest.approx(
            2.0 / 3.0, abs=1e-5
        )

    def test_smae_symmetry(self):
        """sMAE should be symmetric."""
        u = jax.random.normal(jax.random.PRNGKey(0), (1, 64))
        v = jax.random.normal(jax.random.PRNGKey(1), (1, 64))
        assert float(ex.metrics.sMAE(u, v)) == pytest.approx(
            float(ex.metrics.sMAE(v, u)), abs=1e-6
        )

    def test_mae_domain_extent_scaling(self):
        """MAE should scale with domain_extent^D."""
        u_pred = 4.0 * jnp.ones((1, 64))
        u_ref = 2.0 * jnp.ones((1, 64))
        L = 5.0
        mae_L1 = float(ex.metrics.MAE(u_pred, u_ref, domain_extent=1.0))
        mae_L5 = float(ex.metrics.MAE(u_pred, u_ref, domain_extent=L))
        assert mae_L5 == pytest.approx(L * mae_L1, abs=1e-5)


# ===========================================================================
# spatial_norm validation tests
# ===========================================================================


class TestSpatialNormValidation:
    def test_normalized_without_ref_raises(self):
        u = jax.random.normal(jax.random.PRNGKey(0), (1, 64))
        with pytest.raises(ValueError, match="normalized.*requires"):
            ex.metrics.spatial_norm(u, mode="normalized")

    def test_symmetric_without_ref_raises(self):
        u = jax.random.normal(jax.random.PRNGKey(0), (1, 64))
        with pytest.raises(ValueError, match="symmetric.*requires"):
            ex.metrics.spatial_norm(u, mode="symmetric")

    def test_outer_exponent_auto(self):
        """When outer_exponent is None, it should be set to 1/inner_exponent."""
        u = jnp.ones((1, 64))
        # For inner=2, outer should auto-set to 0.5
        # spatial_aggregator with p=2, q=0.5 on all-ones: ((1/64)*64*1)^0.5 = 1.0
        result = float(ex.metrics.spatial_norm(u, inner_exponent=2.0))
        assert result == pytest.approx(1.0, abs=1e-5)
