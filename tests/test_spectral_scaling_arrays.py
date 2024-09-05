import jax
import jax.numpy as jnp
import pytest

import exponax as ex


@pytest.mark.parametrize(
    "num_spatial_dims,num_points", [(D, N) for D in [1, 2, 3] for N in [10, 11]]
)
def test_building_scaling_array_for_norm_compensation(
    num_spatial_dims: int, num_points: int
):
    noise = jax.random.normal(
        jax.random.PRNGKey(0), (1,) + (num_points,) * num_spatial_dims
    )

    noise_hat_norm_backward = jnp.fft.rfftn(
        noise,
        axes=ex.spectral.space_indices(num_spatial_dims),
    )
    noise_hat_norm_forward = jnp.fft.rfftn(
        noise,
        axes=ex.spectral.space_indices(num_spatial_dims),
        norm="forward",
    )

    scaling_array = ex.spectral.build_scaling_array(
        num_spatial_dims,
        num_points,
        mode="norm_compensation",
    )

    noise_hat_norm_backward_scaled = noise_hat_norm_backward / scaling_array

    assert noise_hat_norm_backward_scaled == pytest.approx(noise_hat_norm_forward)


# Mode "reconstruction" is already tested as part of the `test_interpolation.py``
