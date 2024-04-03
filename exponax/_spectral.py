from typing import TypeVar, Union

import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float

D = TypeVar("D")


def build_wavenumbers(
    num_spatial_dims: int,
    num_points: int,
    *,
    indexing: str = "ij",
) -> Float[Array, "D ... (N//2)+1"]:
    """
    Setup an array containing integer coordinates of wavenumbers associated with
    a "num_spatial_dims"-dimensional rfft (real-valued FFT)
    `jax.numpy.fft.rfftn`.

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.
        - `num_points`: The number of points in each spatial dimension.
        - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
          Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**
        - `wavenumbers`: An array of wavenumber integer coordinates, shape
            `(D, ..., (N//2)+1)`.
    """
    right_most_wavenumbers = jnp.fft.rfftfreq(num_points, 1 / num_points)
    other_wavenumbers = jnp.fft.fftfreq(num_points, 1 / num_points)

    wavenumber_list = [
        other_wavenumbers,
    ] * (num_spatial_dims - 1) + [
        right_most_wavenumbers,
    ]

    wavenumbers = jnp.stack(
        jnp.meshgrid(*wavenumber_list, indexing=indexing),
    )

    return wavenumbers


def build_scaled_wavenumbers(
    num_spatial_dims: int,
    domain_extent: float,
    num_points: int,
    *,
    indexing: str = "ij",
) -> Float[Array, "D ... (N//2)+1"]:
    """
    Setup an array containing scaled wavenumbers associated with a
    "num_spatial_dims"-dimensional rfft (real-valued FFT)
    `jax.numpy.fft.rfftn`. Scaling is done by `2 * pi / L`.

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.
        - `domain_extent`: The domain extent.
        - `num_points`: The number of points in each spatial dimension.
        - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
          Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**
        - `wavenumbers`: An array of wavenumber integer coordinates, shape
            `(D, ..., (N//2)+1)`.
    """
    scale = 2 * jnp.pi / domain_extent
    wavenumbers = build_wavenumbers(num_spatial_dims, num_points, indexing=indexing)
    return scale * wavenumbers


def derivative(
    field: Float[Array, "C ... N"],
    domain_extent: float,
    *,
    order: int = 1,
    indexing: str = "ij",
) -> Union[Float[Array, "C D ... (N//2)+1"], Float[Array, "D ... (N//2)+1"]]:
    """
    Perform the spectral derivative of a field. In higher dimensions, this
    defaults to the gradient (the collection of all partial derivatives). In 1d,
    the resulting channel dimension holds the derivative. If the function is
    called with an d-dimensional field which has 1 channel, the result will be a
    d-dimensional field with d channels (one per partial derivative). If the
    field originally had C channels, the result will be a matrix field with C
    rows and d columns.

    Note that applying this operator twice will produce issues at the Nyquist if
    the number of degrees of freedom N is even. For this, consider also using
    the order option.

    **Arguments:**
        - `field`: The field to differentiate, shape `(C, ..., N,)`. `C` can be
            `1` for a scalar field or `D` for a vector field.
        - `L`: The domain extent.
        - `order`: The order of the derivative. Default is `1`.
        - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
          Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**
        - `field_der`: The derivative of the field, shape `(C, D, ...,
          (N//2)+1)` or `(D, ..., (N//2)+1)`.
    """
    channel_shape = field.shape[0]
    spatial_shape = field.shape[1:]
    D = len(spatial_shape)
    N = spatial_shape[0]
    derivative_operator = build_derivative_operator(
        D, domain_extent, N, indexing=indexing
    )
    # # I decided to not use this fix

    # # Required for even N, no effect for odd N
    # derivative_operator_fixed = (
    #     derivative_operator * nyquist_filter_mask(D, N)
    # )
    derivative_operator_fixed = derivative_operator**order

    field_hat = jnp.fft.rfftn(field, axes=space_indices(D))
    if channel_shape == 1:
        # Do not introduce another channel axis
        field_der_hat = derivative_operator_fixed * field_hat
    else:
        # Create a "derivative axis" right after the channel axis
        field_der_hat = field_hat[:, None] * derivative_operator_fixed[None, ...]

    field_der = jnp.fft.irfftn(field_der_hat, s=spatial_shape, axes=space_indices(D))

    return field_der


def make_incompressible(
    field: Float[Array, "D ... N"],
    *,
    indexing: str = "ij",
):
    channel_shape = field.shape[0]
    spatial_shape = field.shape[1:]
    num_spatial_dims = len(spatial_shape)
    if channel_shape != num_spatial_dims:
        raise ValueError(
            f"Expected the number of channels to be {num_spatial_dims}, got {channel_shape}."
        )
    num_points = spatial_shape[0]

    derivative_operator = build_derivative_operator(
        num_spatial_dims, 1.0, num_points, indexing=indexing
    )  # domain_extent does not matter because it will cancel out

    incompressible_field_hat = jnp.fft.rfftn(
        field, axes=space_indices(num_spatial_dims)
    )

    divergence = jnp.sum(
        derivative_operator * incompressible_field_hat, axis=0, keepdims=True
    )

    laplace_operator = build_laplace_operator(derivative_operator)

    inv_laplace_operator = jnp.where(
        laplace_operator == 0,
        1.0,
        1.0 / laplace_operator,
    )

    pseudo_pressure = -inv_laplace_operator * divergence

    pseudo_pressure_garadient = derivative_operator * pseudo_pressure

    incompressible_field_hat = incompressible_field_hat - pseudo_pressure_garadient

    incompressible_field = jnp.fft.irfftn(
        incompressible_field_hat, s=spatial_shape, axes=space_indices(num_spatial_dims)
    )

    return incompressible_field


def build_derivative_operator(
    num_spatial_dims: int,
    domain_extent: float,
    num_points: int,
    *,
    indexing: str = "ij",
) -> Complex[Array, "D ... (N//2)+1"]:
    """
    Setup the derivative operator in Fourier space.

    **Arguments:**
        - `D`: The number of spatial dimensions.
        - `L`: The domain extent.
        - `N`: The number of points in each spatial dimension.
        - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
          Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**
        - `derivative_operator`: The derivative operator, shape `(D, ...,
          N//2+1)`.
    """
    return 1j * build_scaled_wavenumbers(
        num_spatial_dims, domain_extent, num_points, indexing=indexing
    )


def build_laplace_operator(
    derivative_operator: Complex[Array, "D ... (N//2)+1"],
    *,
    order: int = 2,
) -> Complex[Array, "1 ... (N//2)+1"]:
    """
    Given the derivative operator of [`build_derivative_operator`], return the
    Laplace operator.

    **Arguments:**
        - `derivative_operator`: The derivative operator, shape `(D, ...,
          N//2+1)`.
        - `order`: The order of the Laplace operator. Default is `2`.

    **Returns:**
        - `laplace_operator`: The Laplace operator, shape `(1, ..., N//2+1)`.
    """
    if order % 2 != 0:
        raise ValueError("Order must be even.")

    return jnp.sum(derivative_operator**order, axis=0, keepdims=True)


def build_gradient_inner_product_operator(
    derivative_operator: Complex[Array, "D ... (N//2)+1"],
    velocity: Float[Array, "D"],
    *,
    order: int = 1,
) -> Complex[Array, "1 ... (N//2)+1"]:
    """
    Given the derivative operator of [`build_derivative_operator`] and a velocity
    field, return the operator that computes the inner product of the gradient
    with the velocity.

    **Arguments:**
        - `derivative_operator`: The derivative operator, shape `(D, ...,
            N//2+1)`.
        - `velocity`: The velocity field, shape `(D,)`.
        - `order`: The order of the gradient. Default is `1`.

    **Returns:**
        - `operator`: The operator, shape `(1, ..., N//2+1)`.
    """
    if order % 2 != 1:
        raise ValueError("Order must be odd.")

    if velocity.shape != (derivative_operator.shape[0],):
        raise ValueError(
            f"Expected velocity shape to be {derivative_operator.shape[0]}, got {velocity.shape}."
        )

    operator = jnp.einsum(
        "i,i...->...",
        velocity,
        derivative_operator**order,
    )

    # Need to add singleton channel axis
    operator = operator[None, ...]

    # Old form below
    # # Need to move the channel/dimension axis last to enable autobroadcast over
    # # the arbitrary number of spatial axes, Then we can move this singleton axis
    # # back to the front
    # operator = jnp.swapaxes(
    #     jnp.sum(
    #         velocity
    #         * jnp.swapaxes(
    #             derivative_operator**order,
    #             0,
    #             -1,
    #         ),
    #         axis=-1,
    #         keepdims=True,
    #     ),
    #     0,
    #     -1,
    # )

    return operator


def space_indices(num_spatial_dims: int) -> tuple[int, ...]:
    """
    Returns the indices within a field array that correspond to the spatial
    dimensions.

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.

    **Returns:**
        - `indices`: The indices of the spatial dimensions.
    """
    return tuple(range(-num_spatial_dims, 0))


def spatial_shape(num_spatial_dims: int, num_points: int) -> tuple[int, ...]:
    """
    Returns the shape of a spatial field array.

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.
        - `num_points`: The number of points in each spatial dimension.

    **Returns:**
        - `shape`: The shape of the spatial field array.
    """
    return (num_points,) * num_spatial_dims


def wavenumber_shape(num_spatial_dims: int, num_points: int) -> tuple[int, ...]:
    """
    Returns the spatial shape of a field in Fourier space (assuming the usage of
    rfft, `jax.numpy.fft.rfftn`).

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.
        - `num_points`: The number of points in each spatial dimension.

    **Returns:**
        - `shape`: The shape of the spatial field array.
    """
    return (num_points,) * (num_spatial_dims - 1) + (num_points // 2 + 1,)


def low_pass_filter_mask(
    num_spatial_dims: int,
    num_points: int,
    *,
    cutoff: int,
    axis_separate: bool = True,
    indexing: str = "ij",
) -> Bool[Array, "1 ... N"]:
    """
    Create a low-pass filter mask in Fourier space.

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.
        - `num_points`: The number of points in each spatial dimension.
        - `cutoff`: The cutoff wavenumber. This is inclusive.
        - `axis_separate`: Whether to apply the cutoff to each axis separately.
          Default is `True`.
        - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
          Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**
        - `mask`: The low-pass filter mask, shape `(1, ..., N//2+1)`.
    """
    wavenumbers = build_wavenumbers(num_spatial_dims, num_points, indexing=indexing)

    if axis_separate:
        mask = True
        for wn_grid in wavenumbers:
            mask = mask & (jnp.abs(wn_grid) <= cutoff)
    else:
        mask = jnp.linalg.norm(mask, axis=0) <= cutoff

    mask = mask[jnp.newaxis, ...]

    return mask


def nyquist_filter_mask(
    num_spatial_dims: int,
    num_points: int,
) -> Bool[Array, "1 ... N"]:
    """
    Creates mask that if multiplied with a field in Fourier space will remove
    the Nyquist mode.

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.
        - `num_points`: The number of points in each spatial dimension.

    **Returns:**
        - `mask`: The Nyquist filter mask, shape `(1, ..., N//2+1)`.
    """
    if num_points % 2 == 1:
        # Odd number of degrees of freedom (no issue with the Nyquist mode)
        return jnp.ones(
            (1, *wavenumber_shape(num_spatial_dims, num_points)), dtype=bool
        )
    else:
        # Even number of dof (hence the Nyquist only appears in the negative
        # wavenumbers. This is problematic because the rfft in D >=2 has
        # multiple FFTs after the rFFT)
        nyquist_mode = num_points // 2 + 1
        mode_below_nyquist = nyquist_mode - 1
        return low_pass_filter_mask(
            num_spatial_dims,
            num_points,
            cutoff=mode_below_nyquist - 1,
            axis_separate=True,
        )

        # # Todo: Do we need the below?
        # wavenumbers = build_wavenumbers(D, N, scaled=False)
        # mask = True
        # for wn_grid in wavenumbers:
        #     mask = mask & (wn_grid != -mode_below_nyquist)
        # return mask


def build_scaling_array(
    num_spatial_dims: int,
    num_points: int,
    *,
    indexing: str = "ij",
) -> Float[Array, "1 ... (N//2)+1"]:
    """
    Creates an array of the values that would be seen in the result of a
    (real-valued) Fourier transform of a signal of amplitude 1.

    **Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions.
        - `num_points`: The number of points in each spatial dimension.
        - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
          Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**
        - `scaling`: The scaling array, shape `(1, ..., N//2+1)`.
    """
    right_most_wavenumbers = jnp.fft.rfftfreq(num_points, 1 / num_points)
    other_wavenumbers = jnp.fft.fftfreq(num_points, 1 / num_points)

    right_most_scaling = jnp.where(
        right_most_wavenumbers == 0,
        num_points,
        num_points / 2,
    )
    other_scaling = jnp.where(
        other_wavenumbers == 0,
        num_points,
        num_points / 2,
    )

    # If N is even, special treatment for the Nyquist mode
    if num_points % 2 == 0:
        # rfft has the Nyquist mode as positive wavenumber
        right_most_scaling = jnp.where(
            right_most_wavenumbers == num_points // 2,
            num_points,
            right_most_scaling,
        )
        # standard fft has the Nyquist mode as negative wavenumber
        other_scaling = jnp.where(
            other_wavenumbers == -num_points // 2,
            num_points,
            other_scaling,
        )

    scaling_list = [
        other_scaling,
    ] * (num_spatial_dims - 1) + [
        right_most_scaling,
    ]

    scaling = jnp.prod(
        jnp.stack(
            jnp.meshgrid(*scaling_list, indexing=indexing),
        ),
        axis=0,
        keepdims=True,
    )

    return scaling
