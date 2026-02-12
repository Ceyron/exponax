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
