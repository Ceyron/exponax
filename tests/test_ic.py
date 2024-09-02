import jax
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
