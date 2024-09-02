"""
Submodule for timesteppers with normalized dynamics. Those steppers operate on
unit dynamics, i.e., `domain_extent = 1.0` and `dt = 1.0`. As such, the
constitutive coefficients are normalized. One has to supply fewer arguments to
uniquely describe a dyanamics.

Additionally, there are Difficulty steppers that interface the same concept
slightly differently.
"""
from ._convection import DifficultyConvectionStepper, NormalizedConvectionStepper
from ._general_nonlinear import (
    DifficultyGeneralNonlinearStepper,
    NormalizedGeneralNonlinearStepper,
)
from ._gradient_norm import DifficultyGradientNormStepper, NormalizedGradientNormStepper
from ._linear import (
    DifficultyLinearStepper,
    DiffultyLinearStepperSimple,
    NormalizedLinearStepper,
)
from ._polynomial import DifficultyPolynomialStepper, NormalizedPolynomialStepper
from ._utils import (
    denormalize_coefficients,
    denormalize_convection_scale,
    denormalize_gradient_norm_scale,
    denormalize_polynomial_scales,
    extract_normalized_coefficients_from_difficulty,
    extract_normalized_convection_scale_from_difficulty,
    extract_normalized_gradient_norm_scale_from_difficulty,
    normalize_coefficients,
    normalize_convection_scale,
    normalize_gradient_norm_scale,
    normalize_polynomial_scales,
    reduce_normalized_coefficients_to_difficulty,
    reduce_normalized_convection_scale_to_difficulty,
    reduce_normalized_gradient_norm_scale_to_difficulty,
)
from ._vorticity_convection import NormalizedVorticityConvection

__all__ = [
    "DifficultyLinearStepper",
    "DiffultyLinearStepperSimple",
    "DifficultyConvectionStepper",
    "DifficultyGradientNormStepper",
    "DifficultyPolynomialStepper",
    "DifficultyGeneralNonlinearStepper",
    "NormalizedConvectionStepper",
    "NormalizedGeneralNonlinearStepper",
    "NormalizedGradientNormStepper",
    "NormalizedLinearStepper",
    "NormalizedPolynomialStepper",
    "NormalizedVorticityConvection",
    "denormalize_coefficients",
    "denormalize_convection_scale",
    "denormalize_gradient_norm_scale",
    "denormalize_polynomial_scales",
    "normalize_coefficients",
    "normalize_convection_scale",
    "normalize_gradient_norm_scale",
    "normalize_polynomial_scales",
    "reduce_normalized_coefficients_to_difficulty",
    "extract_normalized_coefficients_from_difficulty",
    "reduce_normalized_convection_scale_to_difficulty",
    "extract_normalized_convection_scale_from_difficulty",
    "reduce_normalized_gradient_norm_scale_to_difficulty",
    "extract_normalized_gradient_norm_scale_from_difficulty",
]
