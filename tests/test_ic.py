import jax
import jax.numpy as jnp
import pytest

import exponax as ex


@pytest.mark.parametrize(
    "num_spatial_dims,ic_gen",
    [
        (num_spatial_dims, ic_gen)
        for num_spatial_dims in [1, 2, 3]
        for ic_gen in [
            ex.ic.GaussianRandomField,
            ex.ic.RandomDiscontinuities,
            ex.ic.RandomGaussianBlobs,
            ex.ic.RandomTruncatedFourierSeries,
        ]
    ],
)
def test_instantiate(num_spatial_dims, ic_gen):
    ic_gen(num_spatial_dims)


@pytest.mark.parametrize(
    "num_spatial_dims,ic_gen",
    [
        (num_spatial_dims, ic_gen)
        for num_spatial_dims in [1, 2, 3]
        for ic_gen in [
            ex.ic.GaussianRandomField,
            ex.ic.RandomDiscontinuities,
            ex.ic.RandomGaussianBlobs,
            ex.ic.RandomTruncatedFourierSeries,
        ]
    ],
)
def test_generate(num_spatial_dims, ic_gen):
    num_points = 15
    ic_distribution = ic_gen(num_spatial_dims)
    ic_distribution(num_points, key=jax.random.PRNGKey(0))


@pytest.mark.parametrize(
    "num_spatial_dims,ic_gen",
    [
        (num_spatial_dims, ic_gen)
        for num_spatial_dims in [1, 2, 3]
        for ic_gen in [
            ex.ic.GaussianRandomField,
            ex.ic.RandomDiscontinuities,
            ex.ic.RandomGaussianBlobs,
            ex.ic.RandomTruncatedFourierSeries,
        ]
    ],
)
def test_generate_ic_set(num_spatial_dims, ic_gen):
    num_points = 15
    num_samples = 10
    ic_distribution = ic_gen(num_spatial_dims)
    ex.build_ic_set(
        ic_distribution,
        num_points=num_points,
        num_samples=num_samples,
        key=jax.random.PRNGKey(0),
    )


@pytest.mark.parametrize(
    "num_spatial_dims,ic_gen",
    [
        (num_spatial_dims, ic_gen)
        for num_spatial_dims in [1, 2, 3]
        for ic_gen in [
            ex.ic.GaussianRandomField,
            ex.ic.RandomGaussianBlobs,
            ex.ic.RandomTruncatedFourierSeries,
        ]
    ],
)
def test_ic_output_shape(num_spatial_dims, ic_gen):
    """Output shape should be (1, N, ..., N) with D spatial axes."""
    num_points = 16
    ic_distribution = ic_gen(num_spatial_dims)
    ic = ic_distribution(num_points, key=jax.random.PRNGKey(0))

    expected_shape = (1,) + (num_points,) * num_spatial_dims
    assert ic.shape == expected_shape


def test_ic_determinism():
    """Same key should produce the same IC; different keys should differ."""
    ic_gen = ex.ic.RandomTruncatedFourierSeries(1)
    num_points = 32

    ic_a = ic_gen(num_points, key=jax.random.PRNGKey(42))
    ic_b = ic_gen(num_points, key=jax.random.PRNGKey(42))
    ic_c = ic_gen(num_points, key=jax.random.PRNGKey(99))

    assert ic_a == pytest.approx(ic_b, abs=1e-7)
    assert not jnp.allclose(ic_a, ic_c)


def test_multi_channel_ic():
    """MultiChannelIC should concatenate channels from sub-generators."""
    gen_a = ex.ic.RandomTruncatedFourierSeries(1)
    gen_b = ex.ic.RandomTruncatedFourierSeries(1)
    gen_c = ex.ic.RandomTruncatedFourierSeries(1)

    multi_gen = ex.ic.RandomMultiChannelICGenerator((gen_a, gen_b, gen_c))

    num_points = 32
    ic = multi_gen(num_points, key=jax.random.PRNGKey(0))

    assert ic.shape == (3, num_points)


def test_clamping_ic():
    """ClampingICGenerator output should be within [min_val, max_val]."""
    base_gen = ex.ic.RandomTruncatedFourierSeries(1)
    min_val, max_val = 0.2, 0.8
    clamped_gen = ex.ic.ClampingICGenerator(base_gen, limits=(min_val, max_val))

    num_points = 64
    ic = clamped_gen(num_points, key=jax.random.PRNGKey(0))

    assert float(jnp.min(ic)) >= min_val - 1e-6
    assert float(jnp.max(ic)) <= max_val + 1e-6


def test_scaled_ic():
    """ScaledICGenerator should scale the output by the given factor."""
    base_gen = ex.ic.RandomTruncatedFourierSeries(1, max_one=True)
    scale = 3.0
    scaled_gen = ex.ic.ScaledICGenerator(base_gen, scale=scale)

    num_points = 32
    key = jax.random.PRNGKey(0)
    ic_base = base_gen(num_points, key=key)
    ic_scaled = scaled_gen(num_points, key=key)

    assert ic_scaled == pytest.approx(ic_base * scale, abs=1e-6)


# ===========================================================================
# DiffusedNoise tests
# ===========================================================================


class TestDiffusedNoise:
    @pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
    def test_output_shape(self, num_spatial_dims):
        gen = ex.ic.DiffusedNoise(num_spatial_dims)
        num_points = 16
        ic = gen(num_points, key=jax.random.PRNGKey(0))
        expected_shape = (1,) + (num_points,) * num_spatial_dims
        assert ic.shape == expected_shape

    def test_zero_mean(self):
        gen = ex.ic.DiffusedNoise(1, zero_mean=True)
        ic = gen(64, key=jax.random.PRNGKey(0))
        assert float(jnp.mean(ic)) == pytest.approx(0.0, abs=1e-5)

    def test_std_one(self):
        gen = ex.ic.DiffusedNoise(1, zero_mean=True, std_one=True)
        ic = gen(64, key=jax.random.PRNGKey(0))
        assert float(jnp.std(ic)) == pytest.approx(1.0, abs=1e-5)

    def test_max_one(self):
        gen = ex.ic.DiffusedNoise(1, max_one=True)
        ic = gen(64, key=jax.random.PRNGKey(0))
        assert float(jnp.max(jnp.abs(ic))) == pytest.approx(1.0, abs=1e-5)

    def test_determinism(self):
        gen = ex.ic.DiffusedNoise(1)
        ic_a = gen(32, key=jax.random.PRNGKey(42))
        ic_b = gen(32, key=jax.random.PRNGKey(42))
        assert ic_a == pytest.approx(ic_b, abs=1e-7)

    def test_different_keys_differ(self):
        gen = ex.ic.DiffusedNoise(1)
        ic_a = gen(32, key=jax.random.PRNGKey(0))
        ic_b = gen(32, key=jax.random.PRNGKey(1))
        assert not jnp.allclose(ic_a, ic_b)

    def test_invalid_zero_mean_false_std_one_true(self):
        with pytest.raises(ValueError, match="zero_mean=False.*std_one=True"):
            ex.ic.DiffusedNoise(1, zero_mean=False, std_one=True)

    def test_invalid_std_one_and_max_one(self):
        with pytest.raises(ValueError, match="std_one=True.*max_one=True"):
            ex.ic.DiffusedNoise(1, zero_mean=True, std_one=True, max_one=True)

    def test_custom_intensity(self):
        gen = ex.ic.DiffusedNoise(1, intensity=0.01)
        ic = gen(32, key=jax.random.PRNGKey(0))
        assert ic.shape == (1, 32)

    def test_2d(self):
        gen = ex.ic.DiffusedNoise(2, zero_mean=True, max_one=True)
        ic = gen(16, key=jax.random.PRNGKey(0))
        assert ic.shape == (1, 16, 16)
        assert float(jnp.max(jnp.abs(ic))) == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# SineWaves1d tests
# ===========================================================================


class TestSineWaves1d:
    def test_basic(self):
        sw = ex.ic.SineWaves1d(
            domain_extent=2 * jnp.pi,
            amplitudes=(1.0,),
            wavenumbers=(1.0,),
            phases=(0.0,),
        )
        grid = ex.make_grid(1, 2 * jnp.pi, 64)
        ic = sw(grid)
        expected = jnp.sin(grid)
        assert ic == pytest.approx(expected, abs=1e-5)

    def test_multi_wave(self):
        sw = ex.ic.SineWaves1d(
            domain_extent=2 * jnp.pi,
            amplitudes=(1.0, 0.5),
            wavenumbers=(1.0, 2.0),
            phases=(0.0, 0.0),
        )
        grid = ex.make_grid(1, 2 * jnp.pi, 64)
        ic = sw(grid)
        expected = jnp.sin(grid) + 0.5 * jnp.sin(2 * grid)
        assert ic == pytest.approx(expected, abs=1e-5)

    def test_with_offset(self):
        sw = ex.ic.SineWaves1d(
            domain_extent=2 * jnp.pi,
            amplitudes=(1.0,),
            wavenumbers=(1.0,),
            phases=(0.0,),
            offset=3.0,
        )
        grid = ex.make_grid(1, 2 * jnp.pi, 64)
        ic = sw(grid)
        expected = jnp.sin(grid) + 3.0
        assert ic == pytest.approx(expected, abs=1e-5)

    def test_std_one(self):
        sw = ex.ic.SineWaves1d(
            domain_extent=2 * jnp.pi,
            amplitudes=(2.0,),
            wavenumbers=(1.0,),
            phases=(0.0,),
            std_one=True,
        )
        grid = ex.make_grid(1, 2 * jnp.pi, 64)
        ic = sw(grid)
        assert float(jnp.std(ic)) == pytest.approx(1.0, abs=1e-4)

    def test_max_one(self):
        sw = ex.ic.SineWaves1d(
            domain_extent=2 * jnp.pi,
            amplitudes=(3.0,),
            wavenumbers=(1.0,),
            phases=(0.0,),
            max_one=True,
        )
        grid = ex.make_grid(1, 2 * jnp.pi, 64)
        ic = sw(grid)
        assert float(jnp.max(jnp.abs(ic))) == pytest.approx(1.0, abs=1e-4)

    def test_non_1d_raises(self):
        sw = ex.ic.SineWaves1d(
            domain_extent=1.0,
            amplitudes=(1.0,),
            wavenumbers=(1.0,),
            phases=(0.0,),
        )
        grid_2d = ex.make_grid(2, 1.0, 16)
        with pytest.raises(ValueError, match="1d"):
            sw(grid_2d)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same"):
            ex.ic.SineWaves1d(
                domain_extent=1.0,
                amplitudes=(1.0, 2.0),
                wavenumbers=(1.0,),
                phases=(0.0,),
            )

    def test_offset_and_std_one_raises(self):
        with pytest.raises(ValueError, match="offset.*std_one"):
            ex.ic.SineWaves1d(
                domain_extent=1.0,
                amplitudes=(1.0,),
                wavenumbers=(1.0,),
                phases=(0.0,),
                offset=1.0,
                std_one=True,
            )

    def test_std_one_and_max_one_raises(self):
        with pytest.raises(ValueError, match="std_one.*max_one"):
            ex.ic.SineWaves1d(
                domain_extent=1.0,
                amplitudes=(1.0,),
                wavenumbers=(1.0,),
                phases=(0.0,),
                std_one=True,
                max_one=True,
            )


class TestRandomSineWaves1d:
    def test_basic(self):
        gen = ex.ic.RandomSineWaves1d(1)
        ic = gen(64, key=jax.random.PRNGKey(0))
        assert ic.shape == (1, 64)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1d"):
            ex.ic.RandomSineWaves1d(2)

    def test_determinism(self):
        gen = ex.ic.RandomSineWaves1d(1)
        ic_a = gen(64, key=jax.random.PRNGKey(42))
        ic_b = gen(64, key=jax.random.PRNGKey(42))
        assert ic_a == pytest.approx(ic_b, abs=1e-7)

    def test_max_one(self):
        gen = ex.ic.RandomSineWaves1d(1, max_one=True)
        ic = gen(64, key=jax.random.PRNGKey(0))
        assert float(jnp.max(jnp.abs(ic))) == pytest.approx(1.0, abs=1e-4)

    def test_std_one(self):
        gen = ex.ic.RandomSineWaves1d(1, std_one=True)
        ic = gen(64, key=jax.random.PRNGKey(0))
        assert float(jnp.std(ic)) == pytest.approx(1.0, abs=1e-4)

    def test_gen_ic_fun_returns_sine_waves(self):
        gen = ex.ic.RandomSineWaves1d(1, cutoff=3)
        ic_fun = gen.gen_ic_fun(key=jax.random.PRNGKey(0))
        assert isinstance(ic_fun, ex.ic.SineWaves1d)

    def test_invalid_offset_std_one(self):
        with pytest.raises(ValueError, match="offset.*std_one"):
            ex.ic.RandomSineWaves1d(1, offset_range=(-1.0, 1.0), std_one=True)

    def test_invalid_std_one_max_one(self):
        with pytest.raises(ValueError, match="std_one.*max_one"):
            ex.ic.RandomSineWaves1d(1, std_one=True, max_one=True)


# ===========================================================================
# MultiChannelIC gen_ic_fun tests
# ===========================================================================


class TestMultiChannelICGenIcFun:
    def test_gen_ic_fun_produces_multi_channel_ic(self):
        # Use RandomSineWaves1d which supports gen_ic_fun
        gen_a = ex.ic.RandomSineWaves1d(1)
        gen_b = ex.ic.RandomSineWaves1d(1)
        multi_gen = ex.ic.RandomMultiChannelICGenerator((gen_a, gen_b))

        ic_fun = multi_gen.gen_ic_fun(key=jax.random.PRNGKey(0))
        assert isinstance(ic_fun, ex.ic.MultiChannelIC)

        grid = ex.make_grid(1, 1.0, 32)
        result = ic_fun(grid)
        assert result.shape == (2, 32)

    def test_channels_are_independent(self):
        """Different generators with same key should produce different channels."""
        gen = ex.ic.RandomTruncatedFourierSeries(1)
        multi_gen = ex.ic.RandomMultiChannelICGenerator((gen, gen, gen))
        ic = multi_gen(32, key=jax.random.PRNGKey(0))
        # Channels should differ because keys are split
        assert not jnp.allclose(ic[0], ic[1])
        assert not jnp.allclose(ic[1], ic[2])
