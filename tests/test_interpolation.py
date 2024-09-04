import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

import exponax as ex

FN_DICT_1D = {
    "constant": lambda x: 3.0 * jnp.ones_like(x),
    "simple_sine": lambda x: jnp.sin(2 * jnp.pi * x),
    "simple_cosine": lambda x: jnp.cos(2 * jnp.pi * x),
    "complicated_fn": lambda x: jnp.exp(-3 * (x - 5.0) ** 2),
}


@pytest.mark.parametrize(
    "fn_name, domain_extent, num_points, query_location",
    [
        ("constant", 1.0, 10, jnp.array([0.3])),
        ("simple_sine", 1.0, 10, jnp.array([0.3])),
        ("simple_cosine", 1.0, 10, jnp.array([0.3])),
        ("complicated_fn", 10.0, 100, jnp.array([4.737])),
    ],
)
def test_fourier_interpolator_1d(
    fn_name: str,
    domain_extent: float,
    num_points: int,
    query_location: Float[Array, "1"],
):
    fn = FN_DICT_1D[fn_name]
    grid = ex.make_grid(1, domain_extent, num_points)

    u = fn(grid)

    interpolator = ex.FourierInterpolator(u, domain_extent=domain_extent)

    interpolated_u = interpolator(query_location)
    correct_val = fn(query_location)

    assert interpolated_u == pytest.approx(correct_val)


FN_DICT_2D = {
    "constant": lambda x: 3.0 * jnp.ones_like(x[0:1]),
    "simple_sine_x": lambda x: jnp.sin(2 * jnp.pi * x[0:1]),
    "simple_cosine_x": lambda x: jnp.cos(2 * jnp.pi * x[0:1]),
    "simple_sine_y": lambda x: jnp.sin(2 * jnp.pi * x[1:2]),
    "simple_cosine_y": lambda x: jnp.cos(2 * jnp.pi * x[1:2]),
    "mixed_sine": lambda x: jnp.sin(2 * jnp.pi * x[0:1]) * jnp.sin(6 * jnp.pi * x[1:2]),
    "mixed_cosine": lambda x: jnp.cos(2 * jnp.pi * x[0:1])
    * jnp.cos(6 * jnp.pi * x[1:2]),
    "complicated_fn_x": lambda x: jnp.exp(-3 * (x[0:1] - 5.0) ** 2),
    "complicated_fn_y": lambda x: jnp.exp(-3 * (x[1:2] - 5.0) ** 2),
    "complicated_fn_xy": lambda x: jnp.exp(
        -3 * ((x[0:1] - 5.0) ** 2 + (x[1:2] - 5.0) ** 2)
    ),
}


@pytest.mark.parametrize(
    "fn_name, domain_extent, num_points, query_location",
    [
        ("constant", 1.0, 20, jnp.array([0.3, 0.3])),
        ("simple_sine_x", 1.0, 20, jnp.array([0.3, 0.3])),
        ("simple_cosine_x", 1.0, 20, jnp.array([0.3, 0.3])),
        ("simple_sine_y", 1.0, 20, jnp.array([0.3, 0.3])),
        ("simple_cosine_y", 1.0, 20, jnp.array([0.3, 0.3])),
        ("mixed_sine", 1.0, 20, jnp.array([0.3, 0.3])),
        ("mixed_cosine", 1.0, 20, jnp.array([0.3, 0.3])),
        ("complicated_fn_x", 10.0, 100, jnp.array([4.737, 0.3])),
        ("complicated_fn_y", 10.0, 100, jnp.array([0.3, 4.737])),
        ("complicated_fn_xy", 10.0, 100, jnp.array([4.737, 4.737])),
    ],
)
def test_fourier_interpolator_2d(
    fn_name: str,
    domain_extent: float,
    num_points: int,
    query_location: Float[Array, "2"],
):
    fn = FN_DICT_2D[fn_name]
    grid = ex.make_grid(2, domain_extent, num_points)

    u = fn(grid)

    interpolator = ex.FourierInterpolator(u, domain_extent=domain_extent)

    interpolated_u = interpolator(query_location)
    correct_val = fn(query_location)

    # Looser rel and abs tol because JAX runs in single precision by default
    assert interpolated_u == pytest.approx(correct_val, rel=1e-5, abs=1e-5)


FN_DICT_3D = {
    "constant": lambda x: 3.0 * jnp.ones_like(x[0:1]),
    "simple_sine_x": lambda x: jnp.sin(2 * jnp.pi * x[0:1]),
    "simple_cosine_x": lambda x: jnp.cos(2 * jnp.pi * x[0:1]),
    "simple_sine_y": lambda x: jnp.sin(2 * jnp.pi * x[1:2]),
    "simple_cosine_y": lambda x: jnp.cos(2 * jnp.pi * x[1:2]),
    "simple_sine_z": lambda x: jnp.sin(2 * jnp.pi * x[2:3]),
    "simple_cosine_z": lambda x: jnp.cos(2 * jnp.pi * x[2:3]),
    "mixed_sine": lambda x: jnp.sin(2 * jnp.pi * x[0:1])
    * jnp.sin(6 * jnp.pi * x[1:2])
    * jnp.sin(10 * jnp.pi * x[2:3]),
    "mixed_cosine": lambda x: jnp.cos(2 * jnp.pi * x[0:1])
    * jnp.cos(6 * jnp.pi * x[1:2])
    * jnp.cos(10 * jnp.pi * x[2:3]),
    "complicated_fn_x": lambda x: jnp.exp(-3 * (x[0:1] - 5.0) ** 2),
    "complicated_fn_y": lambda x: jnp.exp(-3 * (x[1:2] - 5.0) ** 2),
    "complicated_fn_z": lambda x: jnp.exp(-3 * (x[2:3] - 5.0) ** 2),
    "complicated_fn_xy": lambda x: jnp.exp(
        -3 * ((x[0:1] - 5.0) ** 2 + (x[1:2] - 5.0) ** 2)
    ),
    "complicated_fn_xz": lambda x: jnp.exp(
        -3 * ((x[0:1] - 5.0) ** 2 + (x[2:3] - 5.0) ** 2)
    ),
    "complicated_fn_yz": lambda x: jnp.exp(
        -3 * ((x[1:2] - 5.0) ** 2 + (x[2:3] - 5.0) ** 2)
    ),
    "complicated_fn_xyz": lambda x: jnp.exp(
        -3 * ((x[0:1] - 5.0) ** 2 + (x[1:2] - 5.0) ** 2 + (x[2:3] - 5.0) ** 2)
    ),
}


@pytest.mark.parametrize(
    "fn_name, domain_extent, num_points, query_location",
    [
        ("constant", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("simple_sine_x", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("simple_cosine_x", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("simple_sine_y", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("simple_cosine_y", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("simple_sine_z", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("simple_cosine_z", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("mixed_sine", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("mixed_cosine", 1.0, 30, jnp.array([0.3, 0.3, 0.3])),
        ("complicated_fn_x", 10.0, 100, jnp.array([4.737, 0.3, 0.3])),
        ("complicated_fn_y", 10.0, 100, jnp.array([0.3, 4.737, 0.3])),
        ("complicated_fn_z", 10.0, 100, jnp.array([0.3, 0.3, 4.737])),
        ("complicated_fn_xy", 10.0, 100, jnp.array([4.737, 4.737, 0.3])),
        ("complicated_fn_xz", 10.0, 100, jnp.array([4.737, 0.3, 4.737])),
        ("complicated_fn_yz", 10.0, 100, jnp.array([0.3, 4.737, 4.737])),
        ("complicated_fn_xyz", 10.0, 100, jnp.array([4.737, 4.737, 4.737])),
    ],
)
def test_fourier_interpolator_3d(
    fn_name: str,
    domain_extent: float,
    num_points: int,
    query_location: Float[Array, "3"],
):
    fn = FN_DICT_3D[fn_name]
    grid = ex.make_grid(3, domain_extent, num_points)

    u = fn(grid)

    interpolator = ex.FourierInterpolator(u, domain_extent=domain_extent)

    interpolated_u = interpolator(query_location)
    correct_val = fn(query_location)

    # Looser rel and abs tol because JAX runs in single precision by default
    assert interpolated_u == pytest.approx(correct_val, rel=1e-5, abs=1e-5)
