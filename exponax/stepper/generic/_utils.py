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

        αᵢ = aᵢ Δt / Lⁱ

    !!! warning
        A consequence of this normalization is that the normalized coefficients
        for high order derivatives will be very small.

    **Arguments:**

    - `coefficients`: coefficients for the linear operator, `coefficients[i]` is
        the coefficient for the `i`-th derivative
    - `domain_extent`: extent of the domain
    - `dt`: time step

    **Returns:**

    - `normalized_coefficients`: normalized coefficients for the linear
        operator, `normalized_coefficients[i]` is the coefficient for the `i`-th
        derivative
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
    then be used again in a genric linear stepper with a physical interface.

        aᵢ = αᵢ Lⁱ / Δt

    **Arguments:**

    - `normalized_coefficients`: coefficients for the linear operator,
        `normalized_coefficients[i]` is the coefficient for the `i`-th
        derivative
    - `domain_extent`: extent of the domain
    - `dt`: time step

    **Returns:**

    - `coefficients`: coefficients for the linear operator, `coefficients[i]` is
        the coefficient for the `i`-th derivative
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
    """
    Normalize the scale (=coefficient) in front of the convection term to be
    used with the normalized generic steppers.

        β₁ = b₁ Δt / L

    **Arguments:**

    - `convection_scale`: scale in front of the convection term, i.e., the `b_1`
        in `𝒩(u) = - b₁ 1/2 (u²)ₓ`
    - `domain_extent`: extent of the domain
    - `dt`: time step

    **Returns:**

    - `normalized_convection_scale`: normalized scale in front of the convection
    """
    normalized_convection_scale = convection_scale * dt / domain_extent
    return normalized_convection_scale


def denormalize_convection_scale(
    normalized_convection_scale: float,
    *,
    domain_extent: float,
    dt: float,
) -> float:
    """
    Denormalize the scale in front of the convection term as it was used in the
    normalized generic steppers to then be used again in a generic stepper with
    a physical interface.

        b₁ = β₁ L / Δt

    **Arguments:**

    - `normalized_convection_scale`: normalized scale in front of the convection
    - `domain_extent`: extent of the domain
    - `dt`: time step

    **Returns:**

    - `convection_scale`: scale in front of the convection term, i.e., the `b_1`
        in `𝒩(u) = - b₁ 1/2 (u²)ₓ`
    """
    convection_scale = normalized_convection_scale / dt * domain_extent
    return convection_scale


def normalize_gradient_norm_scale(
    gradient_norm_scale: float,
    *,
    domain_extent: float,
    dt: float,
) -> float:
    """
    Normalize the scale in front of the gradient norm term to be used with the
    normalized generic steppers.

        β₂ = b₂ Δt / L²

    **Arguments:**

    - `gradient_norm_scale`: scale in front of the gradient norm term, i.e., the
        `b_2` in `𝒩(u) = - b₂ 1/2 ‖∇u‖₂²`
    - `domain_extent`: extent of the domain
    - `dt`: time step

    **Returns:**

    - `normalized_gradient_norm_scale`: normalized scale in front of the
        gradient norm term
    """
    normalized_gradient_norm_scale = (
        gradient_norm_scale * dt / jnp.square(domain_extent)
    )
    return normalized_gradient_norm_scale


def denormalize_gradient_norm_scale(
    normalized_gradient_norm_scale: float,
    *,
    domain_extent: float,
    dt: float,
) -> float:
    """
    Denormalize the scale in front of the gradient norm term as it was used in
    the normalized generic steppers to then be used again in a generic stepper
    with a physical interface.

        b₂ = β₂ L² / Δt

    **Arguments:**

    - `normalized_gradient_norm_scale`: normalized scale in front of the gradient
        norm term
    - `domain_extent`: extent of the domain
    - `dt`: time step

    **Returns:**

    - `gradient_norm_scale`: scale in front of the gradient norm term, i.e., the
        `b_2` in `𝒩(u) = - b₂ 1/2 ‖∇u‖₂²`
    """
    gradient_norm_scale = (
        normalized_gradient_norm_scale / dt * jnp.square(domain_extent)
    )
    return gradient_norm_scale


def normalize_polynomial_scales(
    polynomial_scales: tuple[float, ...],
    *,
    domain_extent: float = None,
    dt: float,
) -> tuple[float, ...]:
    """
    Normalize the polynomial scales to be used with the normalized polynomial
    stepper.

    **Arguments:**

    - `polynomial_scales`: scales for the polynomial operator,
        `polynomial_scales[i]` is the scale for the `i`-th degree polynomial
    - `domain_extent`: extent of the domain (not needed, kept for
        compatibility with other normalization APIs)
    - `dt`: time step

    **Returns:**

    - `normalized_polynomial_scales`: normalized scales for the polynomial
        operator, `normalized_polynomial_scales[i]` is the scale for the `i`-th
        degree polynomial
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
        `normalized_polynomial_scales[i]` is the scale for the `i`-th degree
        polynomial
    - `domain_extent`: extent of the domain (not needed, kept for
        compatibility with other normalization APIs)
    - `dt`: time step

    **Returns:**

    - `polynomial_scales`: scales for the polynomial operator,
    """
    polynomial_scales = tuple(c_n / dt for c_n in normalized_polynomial_scales)
    return polynomial_scales


def reduce_normalized_coefficients_to_difficulty(
    normalized_coefficients: tuple[float, ...],
    *,
    num_spatial_dims: int,
    num_points: int,
) -> tuple[float, ...]:
    """
    Reduce the normalized coefficients for a linear operator to a difficulty
    based interface. This interface is designed to "reduce the intensity of the
    dynamics" at higher resolutions to make emulator learning across resolutions
    comparible. Thereby, it resembles the stability numbers of the most compact
    finite difference scheme of the respective PDE.

        γ₀ = α₀

        γⱼ = αⱼ Nʲ 2ʲ⁻¹ D

    **Arguments:**

    - `normalized_coefficients`: normalized coefficients for the linear
        operator, `normalized_coefficients[i]` is the coefficient for the `i`-th
        derivative
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension

    **Returns:**

    - `difficulty_coefficients`: difficulty coefficients for the linear operator,
        `difficulty_coefficients[i]` is the coefficient for the `i`-th derivative
    """
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
) -> tuple[float, ...]:
    """
    Extract the normalized coefficients for a linear operator from a difficulty
    based interface.

        α₀ = γ₀

        αⱼ = γⱼ / (Nʲ 2ʲ⁻¹ D)

    **Arguments:**

    - `difficulty_coefficients`: difficulty coefficients for the linear operator,
        `difficulty_coefficients[i]` is the coefficient for the `i`-th derivative
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension

    **Returns:**

    - `normalized_coefficients`: normalized coefficients for the linear operator,
        `normalized_coefficients[i]` is the coefficient for the `i`-th derivative
    """
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
) -> float:
    """
    Reduce the normalized convection scale to a difficulty based interface.

        δ₁ = β₁ * M * N * D

    **Arguments:**

    - `normalized_convection_scale`: normalized convection scale, see also
        `exponax.stepper.generic.normalize_convection_scale`
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension
    - `maximum_absolute`: maximum absolute value of the input state the
        resulting stepper is applied to

    **Returns:**

    - `difficulty_convection_scale`: difficulty convection scale
    """
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
) -> float:
    """
    Extract the normalized convection scale from a difficulty based interface.

        β₁ = δ₁ / (M * N * D)

    **Arguments:**

    - `difficulty_convection_scale`: difficulty convection scale
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension
    - `maximum_absolute`: maximum absolute value of the input state the
        resulting stepper is applied to

    **Returns:**

    - `normalized_convection_scale`: normalized convection scale, see also
        `exponax.stepper.generic.normalize_convection_scale`
    """
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
) -> float:
    """
    Reduce the normalized gradient norm scale to a difficulty based interface.

        δ₂ = β₂ * M * N² * D

    **Arguments:**

    - `normalized_gradient_norm_scale`: normalized gradient norm scale, see also
        `exponax.stepper.generic.normalize_gradient_norm_scale`
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension
    - `maximum_absolute`: maximum absolute value of the input state the
        resulting stepper is applied to

    **Returns:**

    - `difficulty_gradient_norm_scale`: difficulty gradient norm scale
    """
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
) -> float:
    """
    Extract the normalized gradient norm scale from a difficulty based interface.

        β₂ = δ₂ / (M * N² * D)

    **Arguments:**

    - `difficulty_gradient_norm_scale`: difficulty gradient norm scale
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension
    - `maximum_absolute`: maximum absolute value of the input state the
        resulting stepper is applied to

    **Returns:**

    - `normalized_gradient_norm_scale`: normalized gradient norm scale, see also
        `exponax.stepper.generic.normalize_gradient_norm_scale`
    """
    normalized_gradient_norm_scale = difficulty_gradient_norm_scale / (
        maximum_absolute * jnp.square(num_points) * num_spatial_dims
    )
    return normalized_gradient_norm_scale


def reduce_normalized_nonlinear_scales_to_difficulty(
    normalized_nonlinear_scales: tuple[float, float, float],
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
) -> tuple[float, float, float]:
    """
    Reduce the normalized nonlinear scales associated with a quadratic, a
    (single-channel) convection term, and a gradient norm term to a difficulty
    based interface.

        δ₀ = β₀

        δ₁ = β₁ * M * N * D

        δ₂ = β₂ * M * N² * D

    **Arguments:**

    - `normalized_nonlinear_scales`: normalized nonlinear scales associated with
        a quadratic, a (single-channel) convection term, and a gradient norm
        term (in this order)
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension
    - `maximum_absolute`: maximum absolute value of the input state the
        resulting stepper is applied to

    **Returns:**

    - `nonlinear_difficulties`: difficulty nonlinear scales associated with a
        quadratic, a (single-channel) convection term, and a gradient norm term
        (in this order)
    """
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
    nonlinear_difficulties: tuple[float, float, float],
    *,
    num_spatial_dims: int,
    num_points: int,
    maximum_absolute: float,
) -> tuple[float, float, float]:
    """
    Extract the normalized nonlinear scales associated with a quadratic, a
    (single-channel) convection term, and a gradient norm term from a difficulty
    based interface.

        β₀ = δ₀

        β₁ = δ₁ / (M * N * D)

        β₂ = δ₂ / (M * N² * D)

    **Arguments:**

    - `nonlinear_difficulties`: difficulty nonlinear scales associated with a
        quadratic, a (single-channel) convection term, and a gradient norm term
        (in this order)
    - `num_spatial_dims`: number of spatial dimensions `d`
    - `num_points`: number of points `N` used to discretize the domain per
        dimension
    - `maximum_absolute`: maximum absolute value of the input state the
        resulting stepper is applied to

    **Returns:**

    - `normalized_nonlinear_scales`: normalized nonlinear scales associated with
        a quadratic, a (single-channel) convection term, and a gradient norm term
        (in this order)
    """
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
