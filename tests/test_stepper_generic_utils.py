"""
Tests for the normalization/denormalization and difficulty reduction/extraction
utilities in exponax.stepper.generic._utils.

These are round-trip tests: normalize then denormalize should recover the
original values, and reduce-to-difficulty then extract-from-difficulty should
recover the original normalized values.
"""

import pytest

from exponax.stepper.generic._utils import (
    denormalize_coefficients,
    denormalize_convection_scale,
    denormalize_gradient_norm_scale,
    denormalize_polynomial_scales,
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_convection_scale_from_difficulty,
    extract_normalized_gradient_norm_scale_from_difficulty,
    extract_normalized_nonlinear_scales_from_difficulty,
    normalize_coefficients,
    normalize_convection_scale,
    normalize_gradient_norm_scale,
    normalize_polynomial_scales,
    reduce_normalized_coefficients_to_difficulty,
    reduce_normalized_convection_scale_to_difficulty,
    reduce_normalized_gradient_norm_scale_to_difficulty,
    reduce_normalized_nonlinear_scales_to_difficulty,
)

# ===========================================================================
# Round-trip: normalize then denormalize
# ===========================================================================


class TestNormalizeDenormalizeRoundTrip:
    @pytest.mark.parametrize(
        "coefficients",
        [
            (0.5,),
            (0.0, -1.0),
            (0.0, 0.0, 0.01),
            (0.0, -0.3, 0.01, 0.001),
        ],
    )
    def test_coefficients(self, coefficients):
        L, dt = 3.0, 0.1
        normalized = normalize_coefficients(coefficients, domain_extent=L, dt=dt)
        recovered = denormalize_coefficients(normalized, domain_extent=L, dt=dt)
        for orig, rec in zip(coefficients, recovered, strict=False):
            assert rec == pytest.approx(orig, abs=1e-10)

    @pytest.mark.parametrize("convection_scale", [0.5, 1.0, -6.0])
    def test_convection_scale(self, convection_scale):
        L, dt = 5.0, 0.05
        normalized = normalize_convection_scale(
            convection_scale, domain_extent=L, dt=dt
        )
        recovered = denormalize_convection_scale(normalized, domain_extent=L, dt=dt)
        assert recovered == pytest.approx(convection_scale, abs=1e-10)

    @pytest.mark.parametrize("gradient_norm_scale", [0.1, 1.0, 3.5])
    def test_gradient_norm_scale(self, gradient_norm_scale):
        L, dt = 2.0, 0.01
        normalized = normalize_gradient_norm_scale(
            gradient_norm_scale, domain_extent=L, dt=dt
        )
        recovered = denormalize_gradient_norm_scale(normalized, domain_extent=L, dt=dt)
        assert recovered == pytest.approx(gradient_norm_scale, abs=1e-10)

    @pytest.mark.parametrize(
        "polynomial_scales",
        [
            (0.1,),
            (0.0, 0.5),
            (0.0, 0.0, -1.0),
        ],
    )
    def test_polynomial_scales(self, polynomial_scales):
        dt = 0.1
        normalized = normalize_polynomial_scales(polynomial_scales, dt=dt)
        recovered = denormalize_polynomial_scales(normalized, dt=dt)
        for orig, rec in zip(polynomial_scales, recovered, strict=False):
            assert rec == pytest.approx(orig, abs=1e-10)


# ===========================================================================
# Round-trip: reduce to difficulty then extract
# ===========================================================================


class TestDifficultyRoundTrip:
    @pytest.mark.parametrize(
        "normalized_coefficients",
        [
            (0.5,),
            (0.0, -0.01),
            (0.0, 0.0, 0.001),
            (0.1, -0.05, 0.002, 0.0001),
        ],
    )
    def test_coefficients_difficulty(self, normalized_coefficients):
        D, N = 2, 32
        difficulty = reduce_normalized_coefficients_to_difficulty(
            normalized_coefficients, num_spatial_dims=D, num_points=N
        )
        recovered = extract_normalized_coefficients_from_difficulty(
            difficulty, num_spatial_dims=D, num_points=N
        )
        for orig, rec in zip(normalized_coefficients, recovered, strict=False):
            assert rec == pytest.approx(orig, abs=1e-10)

    @pytest.mark.parametrize("normalized_scale", [0.01, 0.5, -0.3])
    def test_convection_difficulty(self, normalized_scale):
        D, N, M = 1, 64, 2.0
        difficulty = reduce_normalized_convection_scale_to_difficulty(
            normalized_scale,
            num_spatial_dims=D,
            num_points=N,
            maximum_absolute=M,
        )
        recovered = extract_normalized_convection_scale_from_difficulty(
            difficulty,
            num_spatial_dims=D,
            num_points=N,
            maximum_absolute=M,
        )
        assert recovered == pytest.approx(normalized_scale, abs=1e-10)

    @pytest.mark.parametrize("normalized_scale", [0.001, 0.1, 1.0])
    def test_gradient_norm_difficulty(self, normalized_scale):
        D, N, M = 2, 32, 1.5
        difficulty = reduce_normalized_gradient_norm_scale_to_difficulty(
            normalized_scale,
            num_spatial_dims=D,
            num_points=N,
            maximum_absolute=M,
        )
        recovered = extract_normalized_gradient_norm_scale_from_difficulty(
            difficulty,
            num_spatial_dims=D,
            num_points=N,
            maximum_absolute=M,
        )
        assert recovered == pytest.approx(normalized_scale, abs=1e-10)

    def test_nonlinear_scales_difficulty(self):
        D, N, M = 1, 64, 2.0
        scales = (0.1, 0.05, 0.01)
        difficulty = reduce_normalized_nonlinear_scales_to_difficulty(
            scales,
            num_spatial_dims=D,
            num_points=N,
            maximum_absolute=M,
        )
        recovered = extract_normalized_nonlinear_scales_from_difficulty(
            difficulty,
            num_spatial_dims=D,
            num_points=N,
            maximum_absolute=M,
        )
        for orig, rec in zip(scales, recovered, strict=False):
            assert rec == pytest.approx(orig, abs=1e-10)


# ===========================================================================
# Specific formula tests
# ===========================================================================


class TestNormalizationFormulas:
    def test_coefficient_formula(self):
        """αᵢ = aᵢ * dt / L^i."""
        coeffs = (1.0, 2.0, 3.0)
        L, dt = 4.0, 0.5
        normalized = normalize_coefficients(coeffs, domain_extent=L, dt=dt)
        assert normalized[0] == pytest.approx(1.0 * 0.5 / 1.0)  # a0 * dt / L^0
        assert normalized[1] == pytest.approx(2.0 * 0.5 / 4.0)  # a1 * dt / L^1
        assert normalized[2] == pytest.approx(3.0 * 0.5 / 16.0)  # a2 * dt / L^2

    def test_convection_formula(self):
        """β₁ = b₁ * dt / L."""
        b, L, dt = 6.0, 3.0, 0.1
        result = normalize_convection_scale(b, domain_extent=L, dt=dt)
        assert result == pytest.approx(6.0 * 0.1 / 3.0)

    def test_gradient_norm_formula(self):
        """β₂ = b₂ * dt / L²."""
        b, L, dt = 2.0, 4.0, 0.05
        result = normalize_gradient_norm_scale(b, domain_extent=L, dt=dt)
        assert result == pytest.approx(2.0 * 0.05 / 16.0)

    def test_polynomial_formula(self):
        """Polynomial normalization: cᵢ_norm = cᵢ * dt."""
        scales = (1.0, 2.0, 3.0)
        dt = 0.1
        result = normalize_polynomial_scales(scales, dt=dt)
        for r, expected in zip(result, (0.1, 0.2, 0.3), strict=False):
            assert r == pytest.approx(expected, abs=1e-10)

    def test_difficulty_zeroth_coefficient_passthrough(self):
        """γ₀ = α₀ (zeroth coefficient is passed through unchanged)."""
        coeffs = (0.42, 0.1, 0.01)
        difficulty = reduce_normalized_coefficients_to_difficulty(
            coeffs, num_spatial_dims=2, num_points=32
        )
        assert difficulty[0] == pytest.approx(0.42)
