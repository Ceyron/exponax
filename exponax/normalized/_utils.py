import jax.numpy as jnp


def normalize_coefficients(
    coefficients: tuple[float, ...],
    *,
    domain_extent: float,
    dt: float,
) -> tuple[float, ...]:
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
    normalized_coefficients: tuple[float, ...],
    *,
    domain_extent: float,
    dt: float,
) -> tuple[float, ...]:
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
    normalized_polynomial_scales: tuple[float, ...],
    *,
    domain_extent: float = None,
    dt: float,
) -> tuple[float, ...]:
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


def reduce_normalized_coefficients_to_difficulty(
    normalized_coefficients: tuple[float, ...],
    *,
    num_spatial_dims: int,
    num_points: int,
):
    difficulty_coefficients = list(
        alpha * num_points**j * 2 ** (j - 1) * num_spatial_dims
        for j, alpha in enumerate(normalized_coefficients)
    )
    difficulty_coefficients[0] = normalized_coefficients[0]

    difficulty_coefficients = tuple(difficulty_coefficients)
    return difficulty_coefficients


def extract_normalized_coefficients_from_difficulty(
    difficulty_coefficients: tuple[float, ...],
    *,
    num_spatial_dims: int,
    num_points: int,
):
    normalized_coefficients = list(
        gamma / (num_points**j * 2 ** (j - 1) * num_spatial_dims)
        for j, gamma in enumerate(difficulty_coefficients)
    )
    normalized_coefficients[0] = difficulty_coefficients[0]

    normalized_coefficients = tuple(normalized_coefficients)
    return normalized_coefficients


def reduce_normalized_convection_scale_to_difficulty(
    normalized_convection_scale: float,
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
):
    difficulty_convection_scale = (
        normalized_convection_scale * maximum_absolute * num_points * num_spatial_dims
    )
    return difficulty_convection_scale


def extract_normalized_convection_scale_from_difficulty(
    difficulty_convection_scale: float,
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
):
    normalized_convection_scale = difficulty_convection_scale / (
        maximum_absolute * num_points * num_spatial_dims
    )
    return normalized_convection_scale


def reduce_normalized_gradient_norm_scale_to_difficulty(
    normalized_gradient_norm_scale: float,
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
):
    difficulty_gradient_norm_scale = (
        normalized_gradient_norm_scale
        * maximum_absolute
        * jnp.square(num_points)
        * num_spatial_dims
    )
    return difficulty_gradient_norm_scale


def extract_normalized_gradient_norm_scale_from_difficulty(
    difficulty_gradient_norm_scale: float,
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
):
    normalized_gradient_norm_scale = difficulty_gradient_norm_scale / (
        maximum_absolute * jnp.square(num_points) * num_spatial_dims
    )
    return normalized_gradient_norm_scale


def reduce_normalized_nonlinear_scales_to_difficulty(
    normalized_nonlinear_scales: tuple[float],
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
):
    nonlinear_difficulties = (
        normalized_nonlinear_scales[0],  # Polynomial: normalized == difficulty
        reduce_normalized_convection_scale_to_difficulty(
            normalized_nonlinear_scales[1],
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            maximum_absolute=maximum_absolute,
        ),
        reduce_normalized_gradient_norm_scale_to_difficulty(
            normalized_nonlinear_scales[2],
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            maximum_absolute=maximum_absolute,
        ),
    )
    return nonlinear_difficulties


def extract_normalized_nonlinear_scales_from_difficulty(
    nonlinear_difficulties: tuple[float],
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
):
    normalized_nonlinear_scales = (
        nonlinear_difficulties[0],  # Polynomial: normalized == difficulty
        extract_normalized_convection_scale_from_difficulty(
            nonlinear_difficulties[1],
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            maximum_absolute=maximum_absolute,
        ),
        extract_normalized_gradient_norm_scale_from_difficulty(
            nonlinear_difficulties[2],
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            maximum_absolute=maximum_absolute,
        ),
    )
    return normalized_nonlinear_scales
