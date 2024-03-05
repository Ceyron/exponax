import jax.numpy as jnp


def normalize_coefficients(
    coefficients: tuple[float],
    *,
    domain_extent: float,
    dt: float,
) -> tuple[float]:
    """
    Normalize the coefficients to a linear time stepper to be used with the
    normalized linear stepper.

    **Arguments:**
    - `coefficients`: coefficients for the linear operator, `coefficients[i]` is
        the coefficient for the `i`-th derivative
    - `domain_extent`: extent of the domain
    - `dt`: time step
    """
    normalized_coefficients = tuple(
        c * dt / (domain_extent**i) for i, c in enumerate(coefficients)
    )
    return normalized_coefficients


def denormalize_coefficients(
    normalized_coefficients: tuple[float],
    *,
    domain_extent: float,
    dt: float,
) -> tuple[float]:
    """
    Denormalize the coefficients as they were used in the normalized linear to
    then be used again in a regular linear stepper.

    **Arguments:**
    - `normalized_coefficients`: coefficients for the linear operator,
        `normalized_coefficients[i]` is the coefficient for the `i`-th
        derivative
    - `domain_extent`: extent of the domain
    - `dt`: time step
    """
    coefficients = tuple(
        c_n / dt * domain_extent**i for i, c_n in enumerate(normalized_coefficients)
    )
    return coefficients


def normalize_convection_scale(
    convection_scale: float,
    *,
    domain_extent: float,
    dt: float,
) -> float:
    normalized_convection_scale = convection_scale * dt / domain_extent
    return normalized_convection_scale


def denormalize_convection_scale(
    normalized_convection_scale: float,
    *,
    domain_extent: float,
    dt: float,
) -> float:
    convection_scale = normalized_convection_scale / dt * domain_extent
    return convection_scale


def normalize_gradient_norm_scale(
    gradient_norm_scale: float,
    *,
    domain_extent: float,
    dt: float,
):
    normalized_gradient_norm_scale = (
        gradient_norm_scale * dt / jnp.square(domain_extent)
    )
    return normalized_gradient_norm_scale


def denormalize_gradient_norm_scale(
    normalized_gradient_norm_scale: float,
    *,
    domain_extent: float,
    dt: float,
):
    gradient_norm_scale = (
        normalized_gradient_norm_scale / dt * jnp.square(domain_extent)
    )
    return gradient_norm_scale


def normalize_polynomial_scales(
    polynomial_scales: tuple[float],
    *,
    domain_extent: float = None,
    dt: float,
) -> tuple[float]:
    """
    Normalize the polynomial scales to be used with the normalized polynomial
    stepper.

    **Arguments:**
        - `polynomial_scales`: scales for the polynomial operator,
            `polynomial_scales[i]` is the scale for the `i`-th derivative
        - `domain_extent`: extent of the domain (not needed, kept for
            compatibility with other normalization APIs)
        - `dt`: time step
    """
    normalized_polynomial_scales = tuple(c * dt for c in polynomial_scales)
    return normalized_polynomial_scales


def denormalize_polynomial_scales(
    normalized_polynomial_scales: tuple[float],
    *,
    domain_extent: float = None,
    dt: float,
) -> tuple[float]:
    """
    Denormalize the polynomial scales as they were used in the normalized
    polynomial to then be used again in a regular polynomial stepper.

    **Arguments:**
        - `normalized_polynomial_scales`: scales for the polynomial operator,
            `normalized_polynomial_scales[i]` is the scale for the `i`-th
            derivative
        - `domain_extent`: extent of the domain (not needed, kept for
            compatibility with other normalization APIs)
        - `dt`: time step
    """
    polynomial_scales = tuple(c_n / dt for c_n in normalized_polynomial_scales)
    return polynomial_scales
