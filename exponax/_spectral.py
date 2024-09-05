from itertools import product
from typing import Literal, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float

C = TypeVar("C")
D = TypeVar("D")
N = TypeVar("N")


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
    Setup an array containing **scaled** wavenumbers associated with a
    "num_spatial_dims"-dimensional rfft (real-valued FFT) `jax.numpy.fft.rfftn`.
    Scaling is done by `2 * pi / L`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `domain_extent`: The domain extent.
    - `num_points`: The number of points in each spatial dimension.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
        Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**

    - `wavenumbers`: An array of wavenumber integer coordinates, shape
        `(D, ..., (N//2)+1)`.

    !!! info
        These correctly scaled wavenumbers are used to set up derivative
        operators via `1j * wavenumbers`.
    """
    scale = 2 * jnp.pi / domain_extent
    wavenumbers = build_wavenumbers(num_spatial_dims, num_points, indexing=indexing)
    return scale * wavenumbers


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

    - `num_spatial_dims`: The number of spatial dimensions `d`.
    - `domain_extent`: The size of the domain `L`; in higher dimensions
        the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
    - `num_points`: The number of points `N` used to discretize the
        domain. This **includes** the left boundary point and **excludes** the
        right boundary point. In higher dimensions; the number of points in each
        dimension is the same. Hence, the total number of degrees of freedom is
        `Nᵈ`.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.

    **Returns:**

    - `derivative_operator`: The derivative operator in Fourier space
        (complex-valued array)
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
    Given the derivative operator of
    [`exponax.spectral.build_derivative_operator`], return the Laplace operator.

    In state space:

        Δ = ∇ ⋅ ∇

    And in Fourier space:

        i² k⃗ᵀ k⃗ = - k⃗ᵀ k⃗

    **Arguments:**

    - `derivative_operator`: The derivative operator in Fourier space.
    - `order`: The order of the Laplace operator. Default is `2`. Use a higher
        even number for "higher-order Laplacians". For example, `order=4` will
        return the biharmonic operator (without spatial mixing).

    **Returns:**

    - `laplace_operator`: The Laplace operator in Fourier space.
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
    Given the derivative operator of [`build_derivative_operator`] and a
    velocity vector, return the operator that computes the inner product of the
    gradient with the velocity.

    In state space this is:

        c⃗ ⋅ ∇

    And in Fourier space:

        c⃗ ⋅ i k⃗

    **Arguments:**

    - `derivative_operator`: The derivative operator in Fourier space.
    - `velocity`: The velocity vector, must be an array with one axis with as
        many dimensions as the derivative operator has in its leading axis.
    - `order`: The order of the gradient. Default is `1` which is the "regular
        gradient". Use higher orders for higher-order gradients given in terms
        elementwise products. For example, `order=3` will return `c⃗ ⋅ (∇ ⊙ ∇ ⊙
        ∇)`

    **Returns:**

    - `operator`: The operator in Fourier space.
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

    return operator


def space_indices(num_spatial_dims: int) -> tuple[int, ...]:
    """
    Returns the axes indices within a state array that correspond to the spatial
    axes.

    !!! example
        For a 2D field array, the spatial indices are `(-2, -1)`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.

    **Returns:**

    - `indices`: The indices of the spatial axes.
    """
    return tuple(range(-num_spatial_dims, 0))


def spatial_shape(num_spatial_dims: int, num_points: int) -> tuple[int, ...]:
    """
    Returns the shape of a spatial field array (without its leading channel
    axis). This follows the `Exponax` convention that the resolution is
    indentical in each dimension.

    !!! example
        For a 2D field array with 64 points in each dimension, the spatial shape
        is `(64, 64)`. For a 3D field array with 32 points in each dimension,
        the spatial shape is `(32, 32, 32)`.

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
    `exponax.fft` which internall performs a real-valued fft
    `jax.numpy.fft.rfftn`).

    !!! example
        For a 2D field array with 64 points in each dimension, the wavenumber shape
        is `(64, 33)`. For a 3D field array with 32 points in each dimension,
        the spatial shape is `(32, 32, 17)`. For a 1D field array with 51 points,
        the wavenumber shape is `(26,)`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `num_points`: The number of points in each spatial dimension.

    **Returns:**

    - `shape`: The shape of the spatial axes of a state array in Fourier space.
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
        If `True` (default) the low-pass chunk is a hypercube in Fourier space.
        If `False`, the low-pass chunk is a sphere in Fourier space. Only
        relevant for `num_spatial_dims` >= 2.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
        Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**

    - `mask`: The low-pass filter mask.


    !!! example
        In 1D with 10 points, a cutoff of 3 will produce the mask

        ```python

        array([[ True,  True,  True,  True, False, False]])

        ```

        To better understand this, let's produce the corresponding wavenumbers:

        ```python

        wn = exponax.spectral.build_wavenumbers(1, 10)

        print(wn)

        # array([[0,  1,  2,  3,  4,  5]])

        ```

        There are 6 wavenumbers in total (because this equals `(N//2)+1`), the
        zeroth wavenumber is the mean mode, and then the mask includes the next
        three wavenumbers because its **`cutoff` is inclusive**.
    """
    wavenumbers = build_wavenumbers(num_spatial_dims, num_points, indexing=indexing)

    if axis_separate:
        mask = True
        for wn_grid in wavenumbers:
            mask = mask & (jnp.abs(wn_grid) <= cutoff)
    else:
        mask = jnp.linalg.norm(wavenumbers, axis=0) <= cutoff

    mask = mask[jnp.newaxis, ...]

    return mask


def oddball_filter_mask(
    num_spatial_dims: int,
    num_points: int,
) -> Bool[Array, "1 ... N"]:
    """
    Creates mask that if multiplied with a field in Fourier space remove the
    Nyquist mode if the number of degrees of freedom is even.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `num_points`: The number of points in each spatial dimension.

    **Returns:**

    - `mask`: The oddball filter mask which is `True` for all wavenumbers except
        the Nyquist mode if the number of degrees of freedom is even.

    !!! example
        ```python

        mask_even = exponax.spectral.oddball_filter_mask(1, 6)

        # array([[ True,  True,  True, False]])

        mask_odd = exponax.spectral.oddball_filter_mask(1, 7)

        # array([[ True,  True,  True,  True]])

        ```

        For higher-dimensional examples, see `tests/test_filter_masks.py`.

    !!! info
        For more background on why this is needed, see
        https://www.mech.kth.se/~mattias/simson-user-guide-v4.0.pdf section
        6.2.4 and https://math.mit.edu/~stevenj/fft-deriv.pdf
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
            # The cutoff is **inclusive**
            cutoff=mode_below_nyquist - 1,
            axis_separate=True,
        )


def _build_scaling_array(
    num_spatial_dims: int,
    num_points: int,
    *,
    right_most_scaling_denominator: Literal[2, 1],
    others_scaling_denominator: Literal[2, 1],
    indexing: str = "ij",
):
    """
    Low-Level routine to build scaling arrays, prefer using `build_scaling_array`.
    """
    right_most_wavenumbers = jnp.fft.rfftfreq(num_points, 1 / num_points)
    other_wavenumbers = jnp.fft.fftfreq(num_points, 1 / num_points)

    right_most_scaling = jnp.where(
        right_most_wavenumbers == 0,
        num_points,
        num_points / right_most_scaling_denominator,
    )
    other_scaling = jnp.where(
        other_wavenumbers == 0,
        num_points,
        num_points / others_scaling_denominator,  # Only difference
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


def build_scaling_array(
    num_spatial_dims: int,
    num_points: int,
    *,
    mode: Literal["norm_compensation", "reconstruction", "coef_extraction"],
    indexing: str = "ij",
) -> Float[Array, "1 ... (N//2)+1"]:
    """
    When `exponax.fft` is used, the resulting array in Fourier space represents
    a scaled version of the Fourier coefficients. Use this function to produce
    arrays to counteract this scaling based on the task.

    1. `"norm_compensation"`: The scaling is exactly the scaling the
       `exponax.ifft` applies.
    2. `"reconstruction"`: Technically `"norm_compensation"` should provide an
        array of coefficients that can be used to build a Fourier interpolant
        (i.e., what `exponax.FourierInterpolator` does). However, since
        `exponax.fft` uses the real-valued FFT, there is only half of the
        contribution for the coefficients along the right-most axis. This mode
        provides the scaling to counteract this.
    3. `"coef_extraction"`: Any of the former modes (in higher dimensions) does
        not produce the same coefficients as the amplitude in the physical space
        (because there is a coefficient contribution both in the positive and
        negative wavenumber). For example, if the signal `3 * cos(2x)` was
        discretized on the domain `[0, 2pi]` with 10 points, the amplitude of
        the Fourier coefficient at the 2nd wavenumber would be `3/2` if rescaled
        with mode `"norm_compensation"`. This mode provides the scaling to
        extract the correct coefficients.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `num_points`: The number of points in each spatial dimension.
    - `mode`: The mode of the scaling array. Either `"norm_compensation"`,
        `"reconstruction"`, or `"coef_extraction"`.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
        Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**

    - `scaling`: The scaling array.
    """
    if mode == "norm_compensation":
        return _build_scaling_array(
            num_spatial_dims,
            num_points,
            right_most_scaling_denominator=1,
            others_scaling_denominator=1,
            indexing=indexing,
        )
    elif mode == "reconstruction":
        return _build_scaling_array(
            num_spatial_dims,
            num_points,
            right_most_scaling_denominator=2,
            others_scaling_denominator=1,
            indexing=indexing,
        )
    elif mode == "coef_extraction":
        return _build_scaling_array(
            num_spatial_dims,
            num_points,
            right_most_scaling_denominator=2,
            others_scaling_denominator=2,
            indexing=indexing,
        )
    else:
        raise ValueError("Invalid mode.")


def get_modes_slices(
    num_spatial_dims: int, num_points: int
) -> tuple[tuple[slice, ...], ...]:
    """
    Produces a tuple of tuple of slices corresponding to all positive and
    negative wavenumber blocks found in the representation of a state in Fourier
    space.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `num_points`: The number of points in each spatial dimension.

    **Returns:**

    - `all_modes_slices`: The tuple of tuple of slices. The outer tuple has
        `2^(D-1)` entries if `D` is the number of spatial dimensions. Each inner
        tuple has `D+1` entries.

    !!! example
        In 1D, there is only one block of coefficients in Fourier space; those
        associated with the positive wavenumbers. The additional `slice(None)`
        in the beginning is for the channel axis.

        ```python

        slices = exponax.spectral.get_modes_slices(1, 10)

        print(slices)

        # (

        #     (slice(None), slice(None, 6)),

        # )

        ```

        In 2D, there are two blocks of coefficients; one for the positive
        wavenumbers and one for the negative wavenumbers along the first axis
        (which cannot be halved because the `rfft` already acts on the last, the
        second spatial axis).

        ```python

        slices = exponax.spectral.get_modes_slices(2, 10)

        print(slices)

        # (

        #     (slice(None), slice(None, 5), slice(None, 6)),

        #     (slice(None), slice(-5, None), slice(None, 6)),

        # )

        ```
    """
    is_even = num_points % 2 == 0
    nyquist_mode = num_points // 2
    if is_even:
        left_slice = slice(None, nyquist_mode)
        right_slice = slice(-nyquist_mode, None)
    else:
        left_slice = slice(None, nyquist_mode + 1)
        right_slice = slice(-nyquist_mode, None)

    # Starts with the right-most slice which is associated with the axis over
    # which we apply the rfft
    slices_ = [[slice(None, nyquist_mode + 1)]]
    # All other axes have both positive and negative wavenumbers
    slices_ += [[left_slice, right_slice] for _ in range(num_spatial_dims - 1)]
    all_modes_slices = [
        [
            slice(None),
        ]
        + list(reversed(p))
        for p in product(*slices_)
    ]
    all_modes_slices = tuple([tuple(block_slices) for block_slices in all_modes_slices])
    return all_modes_slices


def fft(
    field: Float[Array, "C ... N"],
    *,
    num_spatial_dims: Optional[int] = None,
) -> Complex[Array, "C ... (N//2)+1"]:
    """
    Perform a **real-valued** FFT of a field. This function is designed for
    states in `Exponax` with a leading channel axis and then one, two, or three
    subsequent spatial axes, **each of the same length** N.

    Only accepts real-valued input fields and performs a real-valued FFT. Hence,
    the last axis of the returned field is of length N//2+1.

    !!! warning
        The argument `num_spatial_dims` can only be correctly inferred if the
        array follows the Exponax convention, e.g., no leading batch axis. For a
        batched operation, use `jax.vmap` on this function.

    **Arguments:**

    - `field`: The state to transform.
    - `num_spatial_dims`: The number of spatial dimensions, i.e., how many
        spatial axes follow the channel axis. Can be inferred from the array if
        it follows the Exponax convention. For example, it is not allowed to
        have a leading batch axis, in such a case use `jax.vmap` on this
        function.

    **Returns:**

    - `field_hat`: The transformed field, shape `(C, ..., N//2+1)`.

    !!! info
        Internally uses `jax.numpy.fft.rfftn` with the default settings for the
        `norm` argument with `norm="backward"`. This means that the forward FFT
        (this function) does not apply any normalization to the result, only the
        [`exponax.ifft`][] function applies normalization. To extract the
        amplitude of the coefficients divide by
        `expoanx.spectral.build_scaling_array`.
    """
    if num_spatial_dims is None:
        num_spatial_dims = field.ndim - 1

    return jnp.fft.rfftn(field, axes=space_indices(num_spatial_dims))


def ifft(
    field_hat: Complex[Array, "C ... (N//2)+1"],
    *,
    num_spatial_dims: Optional[int] = None,
    num_points: Optional[int] = None,
) -> Float[Array, "C ... N"]:
    """
    Perform the inverse **real-valued** FFT of a field. This is the inverse
    operation of `exponax.fft`. This function is designed for states in
    `Exponax` with a leading channel axis and then one, two, or three following
    spatial axes. In state space all spatial axes have the same length N (here
    called `num_points`).

    Requires a complex-valued field in Fourier space with the last axis of
    length N//2+1.

    !!! info
        The number of points (N, or `num_points`) must be provided if the number
        of spatial dimensions is 1. Otherwise, it can be inferred from the shape
        of the field.

    !!! warning
        The argument `num_spatial_dims` can only be correctly inferred if the
        array follows the Exponax convention, e.g., no leading batch axis. For a
        batched operation, use `jax.vmap` on this function.

    **Arguments:**

    - `field_hat`: The transformed field, shape `(C, ..., N//2+1)`.
    - `num_spatial_dims`: The number of spatial dimensions, i.e., how many
        spatial axes follow the channel axis. Can be inferred from the array if
        it follows the Exponax convention. For example, it is not allowed to
        have a leading batch axis, in such a case use `jax.vmap` on this
        function.
    - `num_points`: The number of points in each spatial dimension. Can be
        inferred if `num_spatial_dims` >= 2

    **Returns:**

    - `field`: The state in physical space, shape `(C, ..., N,)`.

    !!! info
        Internally uses `jax.numpy.fft.irfftn` with the default settings for the
        `norm` argument with `norm="backward"`. This means that the forward FFT
        [`exponax.fft`][] function does not apply any normalization to the
        input, only the inverse FFT (this function) applies normalization.
        Hence, if you want to define a state in Fourier space and inversely
        transform it, consider using [`exponax.spectral.build_scaling_array`][]
        to correctly scale the complex values before transforming them back.
    """
    if num_spatial_dims is None:
        num_spatial_dims = field_hat.ndim - 1

    if num_points is None:
        if num_spatial_dims >= 2:
            num_points = field_hat.shape[-2]
        else:
            raise ValueError("num_points must be provided if num_spatial_dims == 1.")
    return jnp.fft.irfftn(
        field_hat,
        s=spatial_shape(num_spatial_dims, num_points),
        axes=space_indices(num_spatial_dims),
    )


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

    !!! warning
        The argument `num_spatial_dims` can only be correctly inferred if the
        array follows the Exponax convention, e.g., no leading batch axis. For a
        batched operation, use `jax.vmap` on this function.

    **Arguments:**

    - `field`: The field to differentiate, shape `(C, ..., N,)`. `C` can be
        `1` for a scalar field or `D` for a vector field.
    - `domain_extent`: The size of the domain `L`; in higher dimensions
        the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
    - `order`: The order of the derivative. Default is `1`.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
        Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**

    - `field_der`: The derivative of the field, shape `(C, D, ...,
        (N//2)+1)` or `(D, ..., (N//2)+1)`.
    """
    channel_shape = field.shape[0]
    spatial_shape = field.shape[1:]
    num_spatial_dims = len(spatial_shape)
    num_points = spatial_shape[0]
    derivative_operator = build_derivative_operator(
        num_spatial_dims, domain_extent, num_points, indexing=indexing
    )
    # # I decided to not use this fix

    # # Required for even N, no effect for odd N
    # derivative_operator_fixed = (
    #     derivative_operator * nyquist_filter_mask(D, N)
    # )
    derivative_operator_fixed = derivative_operator**order

    field_hat = fft(field, num_spatial_dims=num_spatial_dims)
    if channel_shape == 1:
        # Do not introduce another channel axis
        field_der_hat = derivative_operator_fixed * field_hat
    else:
        # Create a "derivative axis" right after the channel axis
        field_der_hat = field_hat[:, None] * derivative_operator_fixed[None, ...]

    field_der = ifft(
        field_der_hat, num_spatial_dims=num_spatial_dims, num_points=num_points
    )

    return field_der


def make_incompressible(
    field: Float[Array, "D ... N"],
    *,
    indexing: str = "ij",
):
    """
    Makes a velocity field incompressible by solving the associated pressure
    Poisson equation and subtract the pressure gradient.

    With the divergence of the velocity field as the right-hand side, solve the
    Poisson equation for pressure `p`

        Δp = - ∇ ⋅ v⃗

    and then correct the velocity field to be incompressible

        v⃗ ← v⃗ - ∇p

    **Arguments:**

    - `field`: The velocity field to make incompressible, shape `(D, ..., N,)`.
        Must have as many channel dimensions as spatial axes.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.

    **Returns:**

    - `incompressible_field`: The incompressible velocity field, shape `(D, ...,
        N,)`.
    """
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

    incompressible_field_hat = fft(field, num_spatial_dims=num_spatial_dims)

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

    incompressible_field = ifft(
        incompressible_field_hat,
        num_spatial_dims=num_spatial_dims,
        num_points=num_points,
    )

    return incompressible_field


def get_spectrum(
    state: Float[Array, "C ... N"],
    *,
    power: bool = True,
) -> Float[Array, "C (N//2)+1"]:
    """
    Preliminary function -> might not be working correctly ... :/

    If passed a vorticity field like produced by
    `exponax.stepper.NavierStokesVorticity`, this will produce the enstrophy
    spectrum.

    Inspired by:
    https://github.com/scaomath/torch-cfd/blob/8c64319272f7660a57c491d823384130823900fe/sfno/visualizations.py#L114

    """
    num_spatial_dims = state.ndim - 1
    num_points = state.shape[-1]

    state_hat = fft(state, num_spatial_dims=num_spatial_dims)
    state_hat_scaled = state_hat / build_scaling_array(
        num_spatial_dims,
        num_points,
        mode="reconstruction",  # because of rfft
    )

    wavenumbers_mesh = build_wavenumbers(num_spatial_dims, num_points)
    wavenumbers_1d = build_wavenumbers(1, num_points)
    wavenumbers_norm = jnp.linalg.norm(wavenumbers_mesh, axis=0, keepdims=True)

    dk = wavenumbers_1d[0, 1] - wavenumbers_1d[0, 0]

    if power:
        magnitude = 0.5 * jnp.abs(state_hat_scaled) ** 2
    else:
        magnitude = jnp.abs(state_hat_scaled)

    spectrum = []

    def power_in_bucket(p, k):
        lower_limit = k - dk / 2
        upper_limit = k + dk / 2
        mask = (wavenumbers_norm[0] >= lower_limit) & (
            wavenumbers_norm[0] < upper_limit
        )
        return jnp.mean(p[mask])

    for k in wavenumbers_1d[0, :]:
        spectrum.append(jax.vmap(power_in_bucket, in_axes=(0, None))(magnitude, k))

    spectrum = jnp.stack(spectrum, axis=-1)
    # spectrum /= jnp.sum(spectrum, axis=-1, keepdims=True)

    return spectrum
