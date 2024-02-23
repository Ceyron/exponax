import jax.numpy as jnp


def normalize_coefficients(
    domain_extent: float,
    dt: float,
    coefficients: tuple[float],
) -> tuple[float]:
    """
    Normalize the coefficients to a linear time stepper to be used with the
    normalized linear stepper.

    **Arguments:**
    - `domain_extent`: extent of the domain
    - `dt`: time step
    - `coefficients`: coefficients for the linear operator, `coefficients[i]` is
        the coefficient for the `i`-th derivative
    """
    normalized_coefficients = tuple(
        c * dt / (domain_extent**i) for i, c in enumerate(coefficients)
    )
    return normalized_coefficients


def denormalize_coefficients(
    domain_extent: float,
    dt: float,
    normalized_coefficients: tuple[float],
) -> tuple[float]:
    """
    Denormalize the coefficients as they were used in the normalized linear to
    then be used again in a regular linear stepper.

    **Arguments:**
    - `domain_extent`: extent of the domain
    - `dt`: time step
    - `normalized_coefficients`: coefficients for the linear operator,
        `normalized_coefficients[i]` is the coefficient for the `i`-th
        derivative
    """
    coefficients = tuple(
        c_n / dt * domain_extent**i for i, c_n in enumerate(normalized_coefficients)
    )
    return coefficients


def normalize_convection_scale(
    domain_extent: float,
    convection_scale: float,
) -> float:
    normalized_convection_scale = convection_scale / domain_extent
    return normalized_convection_scale


def denormalize_convection_scale(
    domain_extent: float,
    normalized_convection_scale: float,
) -> float:
    convection_scale = normalized_convection_scale * domain_extent
    return convection_scale


def normalize_gradient_norm_scale(
    domain_extent: float,
    gradient_norm_scale: float,
):
    normalized_gradient_norm_scale = gradient_norm_scale / jnp.square(domain_extent)
    return normalized_gradient_norm_scale


def denormalize_gradient_norm_scale(
    domain_extent: float,
    normalized_gradient_norm_scale: float,
):
    gradient_norm_scale = normalized_gradient_norm_scale * jnp.square(domain_extent)
    return gradient_norm_scale
