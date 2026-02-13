import jax.numpy as jnp
import pytest

import exponax as ex
from exponax._spectral import build_derivative_operator, fft, ifft
from exponax.nonlin_fun import (
    ConvectionNonlinearFun,
    GradientNormNonlinearFun,
    PolynomialNonlinearFun,
    VorticityConvection2d,
)
from exponax.nonlin_fun._general_nonlinear import GeneralNonlinearFun


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


# ===========================================================================
# Validation error path tests
# ===========================================================================


class TestConvectionValidation:
    def test_multi_channel_conservative_channel_mismatch(self):
        """Multi-channel conservative convection requires channels == spatial dims."""
        N = 32
        deriv_op = build_derivative_operator(2, 1.0, N)
        conv = ConvectionNonlinearFun(
            num_spatial_dims=2,
            num_points=N,
            derivative_operator=deriv_op,
            dealiasing_fraction=2 / 3,
            scale=1.0,
            single_channel=False,
            conservative=True,
        )
        # Give 3-channel input (but spatial dims is 2)
        u_hat = jnp.zeros((3, N, N // 2 + 1), dtype=jnp.complex64)
        with pytest.raises(ValueError, match="channels"):
            conv(u_hat)

    def test_multi_channel_nonconservative_channel_mismatch(self):
        """
        Multi-channel non-conservative convection requires channels ==
        spatial dims.
        """
        N = 32
        deriv_op = build_derivative_operator(2, 1.0, N)
        conv = ConvectionNonlinearFun(
            num_spatial_dims=2,
            num_points=N,
            derivative_operator=deriv_op,
            dealiasing_fraction=2 / 3,
            scale=1.0,
            single_channel=False,
            conservative=False,
        )
        u_hat = jnp.zeros((3, N, N // 2 + 1), dtype=jnp.complex64)
        with pytest.raises(ValueError, match="channels"):
            conv(u_hat)


class TestVorticityConvectionValidation:
    def test_non_2d_raises(self):
        """VorticityConvection2d only supports 2D."""
        N = 16
        deriv_op_1d = build_derivative_operator(1, 1.0, N)
        with pytest.raises(ValueError, match="2"):
            VorticityConvection2d(
                num_spatial_dims=1,
                num_points=N,
                derivative_operator=deriv_op_1d,
                dealiasing_fraction=2 / 3,
            )


class TestGeneralNonlinearFunValidation:
    def test_scale_list_wrong_length(self):
        """GeneralNonlinearFun requires exactly 3 scales."""
        N = 32
        deriv_op = build_derivative_operator(1, 1.0, N)
        with pytest.raises(ValueError, match="3"):
            GeneralNonlinearFun(
                num_spatial_dims=1,
                num_points=N,
                derivative_operator=deriv_op,
                dealiasing_fraction=2 / 3,
                scale_list=(1.0, 2.0),  # only 2
            )


# ===========================================================================
# Multi-channel convection tests
# ===========================================================================


class TestMultiChannelConvection:
    def test_2d_conservative(self):
        """Multi-channel conservative convection in 2D should produce finite results."""
        N = 32
        D = 2
        deriv_op = build_derivative_operator(D, 1.0, N)
        conv = ConvectionNonlinearFun(
            num_spatial_dims=D,
            num_points=N,
            derivative_operator=deriv_op,
            dealiasing_fraction=2 / 3,
            scale=1.0,
            single_channel=False,
            conservative=True,
        )
        # 2-channel input matching 2 spatial dims
        grid = ex.make_grid(D, 1.0, N)
        u = jnp.concatenate(
            [
                jnp.sin(2 * jnp.pi * grid[0:1]),
                jnp.cos(2 * jnp.pi * grid[1:2]),
            ],
            axis=0,
        )
        u_hat = fft(u, num_spatial_dims=D)
        result_hat = conv(u_hat)
        result = ifft(result_hat, num_spatial_dims=D, num_points=N)
        assert result.shape == u.shape
        assert jnp.all(jnp.isfinite(result))

    def test_2d_nonconservative(self):
        """Multi-channel non-conservative convection in 2D."""
        N = 32
        D = 2
        deriv_op = build_derivative_operator(D, 1.0, N)
        conv = ConvectionNonlinearFun(
            num_spatial_dims=D,
            num_points=N,
            derivative_operator=deriv_op,
            dealiasing_fraction=2 / 3,
            scale=1.0,
            single_channel=False,
            conservative=False,
        )
        grid = ex.make_grid(D, 1.0, N)
        u = jnp.concatenate(
            [
                jnp.sin(2 * jnp.pi * grid[0:1]),
                jnp.cos(2 * jnp.pi * grid[1:2]),
            ],
            axis=0,
        )
        u_hat = fft(u, num_spatial_dims=D)
        result_hat = conv(u_hat)
        result = ifft(result_hat, num_spatial_dims=D, num_points=N)
        assert result.shape == u.shape
        assert jnp.all(jnp.isfinite(result))


# ===========================================================================
# Gradient norm additional tests
# ===========================================================================


class TestGradientNormAdditional:
    def test_2d(self):
        """GradientNormNonlinearFun should work in 2D."""
        N = 32
        D = 2
        deriv_op = build_derivative_operator(D, 1.0, N)
        grad_norm = GradientNormNonlinearFun(
            num_spatial_dims=D,
            num_points=N,
            derivative_operator=deriv_op,
            dealiasing_fraction=2 / 3,
            zero_mode_fix=True,
            scale=1.0,
        )
        grid = ex.make_grid(D, 1.0, N)
        u = jnp.sin(2 * jnp.pi * grid[0:1]) * jnp.cos(2 * jnp.pi * grid[1:2])
        u_hat = fft(u, num_spatial_dims=D)
        result_hat = grad_norm(u_hat)
        result = ifft(result_hat, num_spatial_dims=D, num_points=N)
        assert result.shape == u.shape
        assert jnp.all(jnp.isfinite(result))
        # DC mode should be near zero due to zero_mode_fix
        assert float(jnp.abs(result_hat[0, 0, 0])) == pytest.approx(0.0, abs=1e-2)

    def test_without_zero_mode_fix(self):
        """Without zero_mode_fix, DC mode may be nonzero."""
        N = 64
        deriv_op = build_derivative_operator(1, 3.0, N)
        grad_norm = GradientNormNonlinearFun(
            num_spatial_dims=1,
            num_points=N,
            derivative_operator=deriv_op,
            dealiasing_fraction=2 / 3,
            zero_mode_fix=False,
            scale=1.0,
        )
        grid = ex.make_grid(1, 3.0, N)
        u = jnp.sin(2 * jnp.pi * grid / 3.0) + 0.5
        u_hat = fft(u, num_spatial_dims=1)
        result_hat = grad_norm(u_hat)
        # Without fix, DC mode should be nonzero (gradient norm squared is positive)
        assert float(jnp.abs(result_hat[0, 0])) > 1e-6
