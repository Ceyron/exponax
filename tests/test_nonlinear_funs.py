import jax.numpy as jnp
import pytest

import exponax as ex
from exponax._spectral import build_derivative_operator, fft, ifft
from exponax.nonlin_fun import (
    ConvectionNonlinearFun,
    GradientNormNonlinearFun,
    PolynomialNonlinearFun,
)


def test_convection_single_channel():
    """
    For u = sin(2*pi*x/L), the single-channel non-conservative convection
    N(u) = -u * du/dx = -sin(x') * cos(x') * (2*pi/L) = -(pi/L)*sin(2x')
    where x' = 2*pi*x/L. Verify the result in Fourier space matches.
    """
    num_points = 64
    domain_extent = 3.0

    derivative_operator = build_derivative_operator(1, domain_extent, num_points)

    conv_fun = ConvectionNonlinearFun(
        num_spatial_dims=1,
        num_points=num_points,
        derivative_operator=derivative_operator,
        dealiasing_fraction=2 / 3,
        scale=1.0,
        single_channel=True,
        conservative=False,
    )

    grid = ex.make_grid(1, domain_extent, num_points)
    u = jnp.sin(2 * jnp.pi * grid / domain_extent)
    u_hat = fft(u, num_spatial_dims=1)

    result_hat = conv_fun(u_hat)
    result = ifft(result_hat, num_spatial_dims=1, num_points=num_points)

    # N(u) = -u * du/dx = -sin(x') * (2*pi/L)*cos(x')
    # = -(2*pi/L) * sin(x')*cos(x') = -(pi/L) * sin(2x')
    # where x' = 2*pi*x/L
    expected = -(jnp.pi / domain_extent) * jnp.sin(
        2 * 2 * jnp.pi * grid / domain_extent
    )

    assert result == pytest.approx(expected, abs=1e-4)


def test_convection_conservative_vs_nonconservative():
    """
    For smooth single-channel input, conservative d(u^2/2)/dx and
    non-conservative u*du/dx should give the same result.
    """
    num_points = 64
    domain_extent = 4.0

    derivative_operator = build_derivative_operator(1, domain_extent, num_points)

    conv_cons = ConvectionNonlinearFun(
        num_spatial_dims=1,
        num_points=num_points,
        derivative_operator=derivative_operator,
        dealiasing_fraction=2 / 3,
        scale=1.0,
        single_channel=True,
        conservative=True,
    )
    conv_noncons = ConvectionNonlinearFun(
        num_spatial_dims=1,
        num_points=num_points,
        derivative_operator=derivative_operator,
        dealiasing_fraction=2 / 3,
        scale=1.0,
        single_channel=True,
        conservative=False,
    )

    grid = ex.make_grid(1, domain_extent, num_points)
    u = jnp.sin(2 * jnp.pi * grid / domain_extent) * 0.5
    u_hat = fft(u, num_spatial_dims=1)

    result_cons = ifft(conv_cons(u_hat), num_spatial_dims=1, num_points=num_points)
    result_noncons = ifft(
        conv_noncons(u_hat), num_spatial_dims=1, num_points=num_points
    )

    assert result_cons == pytest.approx(result_noncons, abs=1e-5)


def test_gradient_norm_zero_mode_fix():
    """
    With zero_mode_fix=True, the zero Fourier mode of the output should be
    zero, preventing drift in the spatial mean.
    """
    num_points = 64
    domain_extent = 3.0

    derivative_operator = build_derivative_operator(1, domain_extent, num_points)

    grad_norm_fun = GradientNormNonlinearFun(
        num_spatial_dims=1,
        num_points=num_points,
        derivative_operator=derivative_operator,
        dealiasing_fraction=2 / 3,
        zero_mode_fix=True,
        scale=1.0,
    )

    grid = ex.make_grid(1, domain_extent, num_points)
    u = jnp.sin(2 * jnp.pi * grid / domain_extent) + 0.5
    u_hat = fft(u, num_spatial_dims=1)

    result_hat = grad_norm_fun(u_hat)

    # The zero mode (DC component) should be zero
    zero_mode = result_hat[0, 0]
    assert float(jnp.abs(zero_mode)) == pytest.approx(0.0, abs=1e-5)


def test_polynomial_nonlinear_fun():
    """
    For a spatially uniform u = c (constant), polynomial N(u) = c0 + c1*u + c2*u^2
    should give a known constant output.
    """
    num_points = 32
    c0, c1, c2 = 1.0, -2.0, 3.0
    u_val = 0.5

    poly_fun = PolynomialNonlinearFun(
        num_spatial_dims=1,
        num_points=num_points,
        dealiasing_fraction=2 / 3,
        coefficients=(c0, c1, c2),
    )

    # Uniform field
    u = jnp.ones((1, num_points)) * u_val
    u_hat = fft(u, num_spatial_dims=1)

    result_hat = poly_fun(u_hat)
    result = ifft(result_hat, num_spatial_dims=1, num_points=num_points)

    expected_value = c0 + c1 * u_val + c2 * u_val**2
    expected = jnp.ones((1, num_points)) * expected_value

    assert result == pytest.approx(expected, abs=1e-5)


def test_dealiasing_removes_high_modes():
    """
    After dealiasing with the 2/3 rule, modes above 2/3 of the Nyquist
    should be zeroed out.
    """
    num_points = 64
    num_spatial_dims = 1
    nyquist = num_points // 2
    cutoff = int(2 / 3 * nyquist)

    # Create a polynomial fun just to get access to the dealias method
    poly_fun = PolynomialNonlinearFun(
        num_spatial_dims=num_spatial_dims,
        num_points=num_points,
        dealiasing_fraction=2 / 3,
        coefficients=(0.0, 1.0),
    )

    # Create input with energy only in high modes (above cutoff)
    u_hat = jnp.zeros((1, num_points // 2 + 1), dtype=jnp.complex64)
    # Put energy in modes above the dealiasing cutoff
    u_hat = u_hat.at[0, cutoff + 1 :].set(1.0 + 0j)

    dealiased = poly_fun.dealias(u_hat)

    # All modes above cutoff should be zero after dealiasing
    assert dealiased[0, cutoff + 1 :] == pytest.approx(
        jnp.zeros(num_points // 2 + 1 - cutoff - 1), abs=1e-10
    )
