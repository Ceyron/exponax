import jax.numpy as jnp
import pytest

import exponax as ex
from exponax._spectral import (
    build_derivative_operator,
    build_gradient_inner_product_operator,
    build_laplace_operator,
    build_scaled_wavenumbers,
    build_scaling_array,
    fft,
    get_fourier_coefficients,
    ifft,
    make_incompressible,
)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_fft_ifft_roundtrip(num_spatial_dims):
    num_points = 32
    domain_extent = 5.0

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u = jnp.sin(2 * jnp.pi * grid[0:1] / domain_extent)

    u_hat = fft(u, num_spatial_dims=num_spatial_dims)
    u_reconstructed = ifft(
        u_hat, num_spatial_dims=num_spatial_dims, num_points=num_points
    )

    assert u_reconstructed == pytest.approx(u, abs=1e-5)


@pytest.mark.parametrize(
    "num_spatial_dims,derivative_axis",
    [
        (1, 0),
        (2, 0),
        (2, 1),
    ],
)
def test_derivative_of_sine(num_spatial_dims, derivative_axis):
    """Spectral derivative of sin(2*pi*k*x/L) should give (2*pi*k/L)*cos(...)."""
    num_points = 64
    domain_extent = 3.0
    k = 3  # wavenumber

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u = jnp.sin(
        k * 2 * jnp.pi * grid[derivative_axis : derivative_axis + 1] / domain_extent
    )

    u_der = ex.derivative(u, domain_extent, order=1)

    expected = (
        k
        * 2
        * jnp.pi
        / domain_extent
        * jnp.cos(
            k * 2 * jnp.pi * grid[derivative_axis : derivative_axis + 1] / domain_extent
        )
    )

    # For 1D (single channel input), derivative returns shape (D, ..., N) For
    # multi-D, it also returns (D, ..., N). We compare the derivative_axis
    # component.
    assert u_der[derivative_axis] == pytest.approx(expected[0], abs=1e-4)


def test_derivative_second_order():
    """Second derivative of sin should give -k^2 * sin."""
    num_points = 64
    domain_extent = 2 * jnp.pi
    k = 3

    grid = ex.make_grid(1, domain_extent, num_points)
    u = jnp.sin(k * grid)

    u_der2 = ex.derivative(u, domain_extent, order=2)

    expected = -(k**2) * jnp.sin(k * grid)

    assert u_der2 == pytest.approx(expected, abs=1e-3)


def test_laplace_operator_eigenvalues():
    """Laplacian eigenvalue for wavenumber k should be -(2*pi*k/L)^2."""
    num_points = 32
    domain_extent = 4.0

    derivative_operator = build_derivative_operator(1, domain_extent, num_points)
    laplace_op = build_laplace_operator(derivative_operator)

    scaled_wavenumbers = build_scaled_wavenumbers(1, domain_extent, num_points)

    expected_eigenvalues = -(scaled_wavenumbers[0:1] ** 2)

    assert laplace_op == pytest.approx(expected_eigenvalues, abs=1e-10)


def test_make_incompressible_preserves_solenoidal():
    """An already divergence-free field should be preserved by the projection."""
    num_points = 32
    domain_extent = 2 * jnp.pi

    grid = ex.make_grid(2, domain_extent, num_points)
    # Stream function psi = sin(x)*sin(y) gives div-free field:
    # vx = -dpsi/dy = -sin(x)*cos(y), vy = dpsi/dx = cos(x)*sin(y)
    vx = -jnp.sin(grid[0:1]) * jnp.cos(grid[1:2])
    vy = jnp.cos(grid[0:1]) * jnp.sin(grid[1:2])
    velocity = jnp.concatenate([vx, vy], axis=0)

    velocity_proj = make_incompressible(velocity)

    assert velocity_proj == pytest.approx(velocity, abs=1e-5)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_build_derivative_operator_shape(num_spatial_dims):
    num_points = 16
    domain_extent = 1.0

    deriv_op = build_derivative_operator(num_spatial_dims, domain_extent, num_points)

    expected_shape = (
        (num_spatial_dims,)
        + (num_points,) * (num_spatial_dims - 1)
        + (num_points // 2 + 1,)
    )

    assert deriv_op.shape == expected_shape


# ===========================================================================
# Laplace operator validation
# ===========================================================================


class TestLaplaceOperator:
    def test_odd_order_raises(self):
        """Laplace operator should reject odd orders."""
        deriv_op = build_derivative_operator(1, 1.0, 32)
        for order in [1, 3, 5]:
            with pytest.raises(ValueError, match="even"):
                build_laplace_operator(deriv_op, order=order)

    def test_biharmonic_eigenvalues(self):
        """Order=4 eigenvalue for wavenumber k should be (2*pi*k/L)^4."""
        N = 32
        L = 4.0
        deriv_op = build_derivative_operator(1, L, N)
        laplace_4 = build_laplace_operator(deriv_op, order=4)
        scaled_wn = build_scaled_wavenumbers(1, L, N)
        expected = scaled_wn[0:1] ** 4
        assert laplace_4 == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("order", [2, 4, 6])
    def test_even_orders_work(self, order):
        """Even orders should produce valid Laplace operators."""
        deriv_op = build_derivative_operator(1, 1.0, 32)
        laplace_op = build_laplace_operator(deriv_op, order=order)
        assert laplace_op.shape == (1, 17)
        assert jnp.all(jnp.isfinite(laplace_op))


# ===========================================================================
# Higher-order derivatives
# ===========================================================================


class TestHigherOrderDerivatives:
    def test_third_derivative(self):
        """d³/dx³ sin(kx) = -k³ cos(kx)."""
        # Use small N to limit max wavenumber and float32 noise amplification
        # (noise ~ 1e-7, amplification ~ k_max^3 = 16^3 ≈ 4000 → error ~ 4e-4)
        N = 32
        L = 2 * jnp.pi
        k = 1
        grid = ex.make_grid(1, L, N)
        u = jnp.sin(k * grid)
        u_der3 = ex.derivative(u, L, order=3)
        expected = -(k**3) * jnp.cos(k * grid)
        assert u_der3 == pytest.approx(expected, abs=1e-3)

    def test_fourth_derivative(self):
        """d⁴/dx⁴ sin(kx) = k⁴ sin(kx)."""
        # Float32 noise amplified by k_max^4 = 16^4 ≈ 65000 → error ~ 7e-3
        N = 32
        L = 2 * jnp.pi
        k = 1
        grid = ex.make_grid(1, L, N)
        u = jnp.sin(k * grid)
        u_der4 = ex.derivative(u, L, order=4)
        expected = (k**4) * jnp.sin(k * grid)
        assert u_der4 == pytest.approx(expected, abs=0.01)


# ===========================================================================
# make_incompressible additional tests
# ===========================================================================


class TestMakeIncompressible:
    def test_removes_divergence_2d(self):
        """Projection should remove divergence from a compressible field."""
        N = 64
        L = 2 * jnp.pi
        grid = ex.make_grid(2, L, N)
        # Compressible field: vx = sin(x), vy = sin(y)
        # div(v) = cos(x) + cos(y) != 0
        vx = jnp.sin(grid[0:1])
        vy = jnp.sin(grid[1:2])
        velocity = jnp.concatenate([vx, vy], axis=0)

        velocity_proj = make_incompressible(velocity)

        # Compute divergence: ∂vx/∂x + ∂vy/∂y
        grad_vx = ex.derivative(velocity_proj[0:1], L, order=1)  # (2, N, N)
        grad_vy = ex.derivative(velocity_proj[1:2], L, order=1)  # (2, N, N)
        divergence = grad_vx[0:1] + grad_vy[1:2]  # ∂vx/∂x + ∂vy/∂y
        assert divergence == pytest.approx(jnp.zeros_like(divergence), abs=1e-5)

    def test_3d(self):
        """make_incompressible should work for 3D velocity fields."""
        N = 16
        L = 2 * jnp.pi
        grid = ex.make_grid(3, L, N)
        # Compressible 3D field
        vx = jnp.sin(grid[0:1])
        vy = jnp.cos(grid[1:2])
        vz = jnp.sin(grid[2:3])
        velocity = jnp.concatenate([vx, vy, vz], axis=0)

        velocity_proj = make_incompressible(velocity)
        assert velocity_proj.shape == (3, N, N, N)
        assert jnp.all(jnp.isfinite(velocity_proj))

        # Compute divergence: ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
        grad_vx = ex.derivative(velocity_proj[0:1], L, order=1)  # (3, N, N, N)
        grad_vy = ex.derivative(velocity_proj[1:2], L, order=1)
        grad_vz = ex.derivative(velocity_proj[2:3], L, order=1)
        divergence = grad_vx[0:1] + grad_vy[1:2] + grad_vz[2:3]
        assert divergence == pytest.approx(jnp.zeros_like(divergence), abs=1e-4)

    def test_idempotent(self):
        """Applying projection twice should give same result as once."""
        N = 32
        L = 2 * jnp.pi
        grid = ex.make_grid(2, L, N)
        vx = jnp.sin(grid[0:1]) + jnp.cos(2 * grid[1:2])
        vy = jnp.cos(grid[0:1]) - jnp.sin(3 * grid[1:2])
        velocity = jnp.concatenate([vx, vy], axis=0)

        v_proj1 = make_incompressible(velocity)
        v_proj2 = make_incompressible(v_proj1)
        assert v_proj2 == pytest.approx(v_proj1, abs=1e-5)


# ===========================================================================
# FFT multi-channel roundtrip
# ===========================================================================


class TestFFTMultiChannel:
    @pytest.mark.parametrize("num_channels", [2, 3, 5])
    def test_roundtrip(self, num_channels):
        """FFT/IFFT roundtrip should preserve multi-channel fields."""
        N = 32
        L = 5.0
        grid = ex.make_grid(1, L, N)
        u = jnp.concatenate(
            [jnp.sin((i + 1) * 2 * jnp.pi * grid / L) for i in range(num_channels)],
            axis=0,
        )
        u_hat = fft(u, num_spatial_dims=1)
        u_rec = ifft(u_hat, num_spatial_dims=1, num_points=N)
        assert u_rec == pytest.approx(u, abs=1e-5)

    def test_2d_multi_channel(self):
        """Multi-channel 2D FFT roundtrip."""
        N = 16
        L = 1.0
        grid = ex.make_grid(2, L, N)
        u = jnp.concatenate(
            [
                jnp.sin(2 * jnp.pi * grid[0:1] / L),
                jnp.cos(2 * jnp.pi * grid[1:2] / L),
            ],
            axis=0,
        )
        u_hat = fft(u, num_spatial_dims=2)
        u_rec = ifft(u_hat, num_spatial_dims=2, num_points=N)
        assert u_rec == pytest.approx(u, abs=1e-5)


# ===========================================================================
# Poisson higher-order tests
# ===========================================================================


class TestPoissonHigherOrder:
    def test_biharmonic_1d(self):
        """Poisson with order=4 solves -Δ²u = f, so u = -f/k⁴ for f=sin(kx)."""
        L = 2 * jnp.pi
        N = 64
        k = 3
        grid = ex.make_grid(1, L, N)
        rhs = jnp.sin(k * grid)
        poisson = ex.poisson.Poisson(1, L, N, order=4)
        u = poisson(rhs)
        # Solver applies u_hat = -inv_operator * f_hat
        # For order=4: laplace_eigenvalue = k^4 (positive), so u = -sin(kx)/k^4
        analytical = -jnp.sin(k * grid) / k**4
        assert u == pytest.approx(analytical, abs=1e-5)

    def test_poisson_multi_mode(self):
        """Poisson with multiple active Fourier modes."""
        L = 2 * jnp.pi
        N = 64
        grid = ex.make_grid(1, L, N)
        # f = sin(x) + 2*sin(3x)
        rhs = jnp.sin(grid) + 2 * jnp.sin(3 * grid)
        poisson = ex.poisson.Poisson(1, L, N)
        u = poisson(rhs)
        # u = sin(x)/1² + 2*sin(3x)/3² = sin(x) + 2/9*sin(3x)
        analytical = jnp.sin(grid) / 1**2 + 2 * jnp.sin(3 * grid) / 3**2
        assert u == pytest.approx(analytical, abs=1e-5)


# ===========================================================================
# Gradient inner product operator
# ===========================================================================


class TestGradientInnerProductOperator:
    def test_even_order_raises(self):
        """Gradient inner product operator requires odd order."""
        deriv_op = build_derivative_operator(1, 1.0, 32)
        velocity = jnp.array([1.0])
        with pytest.raises(ValueError, match="odd"):
            build_gradient_inner_product_operator(deriv_op, velocity, order=2)

    def test_wrong_velocity_shape_raises(self):
        """Velocity shape must match number of spatial dimensions."""
        deriv_op = build_derivative_operator(2, 1.0, 16)
        velocity = jnp.array([1.0, 2.0, 3.0])  # 3 elements but 2D operator
        with pytest.raises(ValueError, match="velocity"):
            build_gradient_inner_product_operator(deriv_op, velocity)

    def test_advection_1d(self):
        """c * d/dx of sin(kx) = c * k * cos(kx)."""
        N = 32
        L = 2 * jnp.pi
        c = 2.0
        k = 1
        deriv_op = build_derivative_operator(1, L, N)
        velocity = jnp.array([c])
        operator = build_gradient_inner_product_operator(deriv_op, velocity)

        grid = ex.make_grid(1, L, N)
        u = jnp.sin(k * grid)
        u_hat = fft(u, num_spatial_dims=1)
        result_hat = operator * u_hat
        result = ifft(result_hat, num_spatial_dims=1, num_points=N)
        expected = c * k * jnp.cos(k * grid)
        assert result == pytest.approx(expected, abs=1e-4)


# ===========================================================================
# Scaling array and Fourier coefficient extraction
# ===========================================================================


class TestBuildScalingArray:
    def test_invalid_mode_raises(self):
        """Invalid scaling mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            build_scaling_array(1, 32, mode="nonexistent_mode")


class TestGetFourierCoefficients:
    def test_no_scaling_compensation(self):
        """With scaling_compensation_mode=None, should return raw FFT output."""
        N = 32
        L = 2 * jnp.pi
        grid = ex.make_grid(1, L, N)
        u = jnp.sin(grid)
        raw_hat = fft(u, num_spatial_dims=1)
        coefficients = get_fourier_coefficients(
            u, scaling_compensation_mode=None, round=None
        )
        assert coefficients == pytest.approx(raw_hat, abs=1e-6)

    def test_no_rounding(self):
        """With round=None, coefficients should not be rounded."""
        N = 32
        L = 2 * jnp.pi
        grid = ex.make_grid(1, L, N)
        u = jnp.sin(grid) + 0.5 * jnp.cos(3 * grid)
        coefs_rounded = get_fourier_coefficients(u, round=5)
        coefs_unrounded = get_fourier_coefficients(u, round=None)
        # Rounded version should have many exact zeros where unrounded has tiny values
        num_zeros_rounded = int(jnp.sum(coefs_rounded == 0))
        num_zeros_unrounded = int(jnp.sum(coefs_unrounded == 0))
        assert num_zeros_rounded >= num_zeros_unrounded

    def test_coef_extraction_mode_reads_amplitudes(self):
        """In coef_extraction mode, amplitude of sin(kx) should be readable."""
        N = 32
        L = 2 * jnp.pi
        grid = ex.make_grid(1, L, N)
        u = 2.0 * jnp.sin(3 * grid)
        coefs = get_fourier_coefficients(u, scaling_compensation_mode="coef_extraction")
        # The amplitude of the k=3 mode should be approximately 2.0
        assert float(jnp.abs(coefs[0, 3])) == pytest.approx(2.0, abs=0.1)


# ===========================================================================
# ifft edge cases
# ===========================================================================


class TestIFFTEdgeCases:
    def test_1d_without_num_points_raises(self):
        """1D ifft without num_points should raise ValueError."""
        u_hat = jnp.zeros((1, 17), dtype=jnp.complex64)
        with pytest.raises(ValueError, match="num_points"):
            ifft(u_hat, num_spatial_dims=None, num_points=None)

    def test_2d_infers_num_points(self):
        """2D ifft should infer num_points from the second-to-last axis."""
        N = 16
        grid = ex.make_grid(2, 1.0, N)
        u = jnp.sin(2 * jnp.pi * grid[0:1])
        u_hat = fft(u, num_spatial_dims=2)
        # Calling without num_points should work for 2D
        u_rec = ifft(u_hat, num_spatial_dims=None, num_points=None)
        assert u_rec == pytest.approx(u, abs=1e-5)


# ===========================================================================
# Multi-channel derivative
# ===========================================================================


class TestMultiChannelDerivative:
    def test_2channel_2d_derivative_shape(self):
        """Derivative of a 2-channel 2D field should give (2, 2, N, N) shape."""
        N = 16
        L = 1.0
        grid = ex.make_grid(2, L, N)
        u = jnp.concatenate(
            [
                jnp.sin(2 * jnp.pi * grid[0:1]),
                jnp.cos(2 * jnp.pi * grid[1:2]),
            ],
            axis=0,
        )  # shape (2, N, N)
        u_der = ex.derivative(u, L, order=1)
        # For C=2, D=2: shape should be (C, D, N, N) = (2, 2, N, N)
        assert u_der.shape == (2, 2, N, N)
        assert jnp.all(jnp.isfinite(u_der))

    def test_2channel_1d_derivative(self):
        """Derivative of a 2-channel 1D field."""
        N = 32
        L = 2 * jnp.pi
        grid = ex.make_grid(1, L, N)
        u = jnp.concatenate(
            [
                jnp.sin(grid),
                jnp.cos(2 * grid),
            ],
            axis=0,
        )  # shape (2, N)
        u_der = ex.derivative(u, L, order=1)
        # shape: (2, 1, N) — 2 channels, 1 spatial dim
        assert u_der.shape == (2, 1, N)
        # d/dx sin(x) = cos(x)
        assert u_der[0, 0] == pytest.approx(jnp.cos(grid[0]), abs=1e-4)
        # d/dx cos(2x) = -2*sin(2x)
        assert u_der[1, 0] == pytest.approx(-2 * jnp.sin(2 * grid[0]), abs=1e-4)


# ===========================================================================
# make_incompressible channel validation
# ===========================================================================


class TestMakeIncompressibleValidation:
    def test_channel_mismatch_raises(self):
        """make_incompressible should reject fields where channels != spatial dims."""
        # 3 channels but 2 spatial dims
        field = jnp.zeros((3, 16, 16))
        with pytest.raises(ValueError, match="channels"):
            make_incompressible(field)


# ===========================================================================
# Poisson solver validation
# ===========================================================================


class TestPoissonValidation:
    def test_wrong_shape_raises(self):
        """Poisson solver should reject input with wrong spatial shape."""
        poisson = ex.poisson.Poisson(1, 2 * jnp.pi, 64)
        wrong_rhs = jnp.ones((1, 32))  # Expected (1, 64)
        with pytest.raises(ValueError, match="Shape"):
            poisson(wrong_rhs)
