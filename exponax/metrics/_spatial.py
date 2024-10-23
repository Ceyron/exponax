from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def spatial_aggregator(
    state_no_channel: Float[Array, "... N"],
    *,
    num_spatial_dims: Optional[int] = None,
    domain_extent: float = 1.0,
    num_points: Optional[int] = None,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
) -> float:
    """
    Aggregate over the spatial axes of a (channel-less) state tensor to get a
    *consistent* counterpart to a functional L^p norm in the continuous case.

    Assuming the `Exponax` convention that the domain is always the scaled
    hypercube `Ω = (0, L)ᴰ` (with `L = domain_extent`) and each spatial
    dimension being discretized uniformly into `N` points (i.e., there are `Nᴰ`
    points in total), and the left boundary is considered a degree of freedom,
    and the right is not, there is the following relation between a continuous
    function `u(x)` and its discretely sampled counterpart `uₕ`

        ‖ u(x) ‖_Lᵖ(Ω) = (∫_Ω |u(x)|ᵖ dx)^(1/p) ≈ ( (L/N)ᴰ ∑ᵢ|uᵢ|ᵖ )^(1/p)

    where the summation `∑ᵢ` must be understood as a sum over all `Nᴰ` points
    across all spatial dimensions. The `inner_exponent` corresponds to `p` in
    the above formula. This function also allows setting the outer exponent `q`
    which via

        ( (L/N)ᴰ ∑ᵢ|uᵢ|ᵖ )^q

    If it is not specified, it is set to `q = 1/p` to get a valid norm.

    !!! tip
        To apply this function to a state tensor with a leading channel axis,
        use `jax.vmap`.

    **Arguments:**

    - `state_no_channel`: The state tensor **without a leading channel
        axis**.
    - `num_spatial_dims`: The number of spatial dimensions. If not specified,
        it is inferred from the number of axes in `state_no_channel`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `num_points`: The number of points `N` in each spatial dimension. If not
        specified, it is inferred from the last axis of `state_no_channel`.
    - `inner_exponent`: The exponent `p` in the L^p norm.
    - `outer_exponent`: The exponent `q` the result after aggregation is raised
        to. If not specified, it is set to `q = 1/p`.

    !!! warning
        To get a truly consistent counterpart to the continuous norm, the
        `domain_extent` must be set. This is relevant to compare performance
        across domain sizes. However, if this is just used as a training
        objective, the `domain_extent` can be set to `1.0` since it only
        contributes a multiplicative factor.

    !!! info
        The approximation to the continuous integral is of the following form:
            - **Exact** if the state is bandlimited.
            - **Exponentially linearly convergent** if the state is smooth. It
                is converged once the state becomes effectively bandlimited
                under `num_points`.
            - **Polynomially linear** in all other cases.
    """
    if num_spatial_dims is None:
        num_spatial_dims = state_no_channel.ndim
    if num_points is None:
        num_points = state_no_channel.shape[-1]

    if outer_exponent is None:
        outer_exponent = 1 / inner_exponent

    scale = (domain_extent / num_points) ** num_spatial_dims

    aggregated = jnp.sum(jnp.abs(state_no_channel) ** inner_exponent)

    return (scale * aggregated) ** outer_exponent


def spatial_norm(
    state: Float[Array, "C ... N"],
    state_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    mode: Literal["absolute", "normalized", "symmetric"] = "absolute",
    domain_extent: float = 1.0,
    inner_exponent: float = 2.0,
    outer_exponent: Optional[float] = None,
) -> float:
    """
    Compute the conistent counterpart of the `Lᴾ` functional norm.

    See [`exponax.metrics.spatial_aggregator`][] for more details. This function
    sums over the channel axis **after aggregation**. If you need more low-level
    control, consider using [`exponax.metrics.spatial_aggregator`][] directly.

    This function allows providing a second state (`state_ref`) to compute
    either the absolute, normalized, or symmetric difference. The `"absolute"`
    mode computes

        (‖uₕ - uₕʳ‖_L^p(Ω))^(q*p)

    while the `"normalized"` mode computes

        (‖uₕ - uₕʳ‖_L^p(Ω))^(q*p) / ((‖uₕʳ‖_L^p(Ω))^(q*p))

    and the `"symmetric"` mode computes

        2 * (‖uₕ - uₕʳ‖_L^p(Ω))^(q*p) / ((‖uₕ‖_L^p(Ω))^(q*p) + (‖uₕʳ‖_L^p(Ω))^(q*p))

    In either way, the channels are summed **after** the aggregation. The
    `inner_exponent` corresponds to `p` in the above formulas. The
    `outer_exponent` corresponds to `q`. If it is not specified, it is set to `q
    = 1/p` to get a valid norm.

    !!! tip
        To operate on states with a leading batch axis, use `jax.vmap`. Then the
        batch axis can be reduced, e.g., by `jnp.mean`. As a helper for this,
        [`exponax.metrics.mean_metric`][] is provided.


    **Arguments:**

    - `state`: The state tensor. Must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `state_ref`: The reference state tensor. Must have the same shape as
        `state`. If not specified, only the absolute norm of `state` is
        computed.
    - `mode`: The mode of the norm. Either `"absolute"`, `"normalized"`, or
        `"symmetric"`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`.
    - `inner_exponent`: The exponent `p` in the L^p norm.
    - `outer_exponent`: The exponent `q` the result after aggregation is raised
        to. If not specified, it is set to `q = 1/p`.
    """
    if state_ref is None:
        if mode == "normalized":
            raise ValueError("mode 'normalized' requires state_ref")
        if mode == "symmetric":
            raise ValueError("mode 'symmetric' requires state_ref")
        diff = state
    else:
        diff = state - state_ref

    diff_norm_per_channel = jax.vmap(
        lambda s: spatial_aggregator(
            s,
            domain_extent=domain_extent,
            inner_exponent=inner_exponent,
            outer_exponent=outer_exponent,
        ),
    )(diff)

    if mode == "normalized":
        ref_norm_per_channel = jax.vmap(
            lambda r: spatial_aggregator(
                r,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
            ),
        )(state_ref)
        normalized_diff_per_channel = diff_norm_per_channel / ref_norm_per_channel
        norm_per_channel = normalized_diff_per_channel
    elif mode == "symmetric":
        state_norm_per_channel = jax.vmap(
            lambda s: spatial_aggregator(
                s,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
            ),
        )(state)
        ref_norm_per_channel = jax.vmap(
            lambda r: spatial_aggregator(
                r,
                domain_extent=domain_extent,
                inner_exponent=inner_exponent,
                outer_exponent=outer_exponent,
            ),
        )(state_ref)
        symmetric_diff_per_channel = (
            2 * diff_norm_per_channel / (state_norm_per_channel + ref_norm_per_channel)
        )
        norm_per_channel = symmetric_diff_per_channel
    else:
        norm_per_channel = diff_norm_per_channel

    return jnp.sum(norm_per_channel)


def MAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the mean absolute error (MAE) between two states.

        ∑_(channels) ∑_(space) (L/N)ᴰ |uₕ - uₕʳ|

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        ‖ u - uʳ ‖_L¹(Ω) = ∫_Ω |u(x) - uʳ(x)| dx

    The channel axis is summed **after** the aggregation.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
        If not specified, the MAE is computed against zero, i.e., the norm of
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor.
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
    )


def nMAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the normalized mean absolute error (nMAE) between two states.

        ∑_(channels) [∑_(space) (L/N)ᴰ |uₕ - uₕʳ| / ∑_(space) (L/N)ᴰ |uₕʳ|]

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        ‖ u - uʳ ‖_L¹(Ω) / ‖ uʳ ‖_L¹(Ω) = ∫_Ω |u(x) - uʳ(x)| dx / ∫_Ω |uʳ(x)| dx

    The channel axis is summed **after** the aggregation.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor.
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
    )


def sMAE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the symmetric mean absolute error (sMAE) between two states.

        ∑_(channels) [2 ∑_(space) (L/N)ᴰ |uₕ - uₕʳ| / (∑_(space) (L/N)ᴰ |uₕ| + ∑_(space) (L/N)ᴰ |uₕʳ|)]

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        2 ∫_Ω |u(x) - uʳ(x)| dx / (∫_Ω |u(x)| dx + ∫_Ω |uʳ(x)| dx)

    The channel axis is summed **after** the aggregation.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! info
        This symmetric metric is bounded between 0 and C with C being the number
        of channels.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor.
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="symmetric",
        domain_extent=domain_extent,
        inner_exponent=1.0,
        outer_exponent=1.0,
    )


def MSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the mean squared error (MSE) between two states.

        ∑_(channels) ∑_(space) (L/N)ᴰ |uₕ - uₕʳ|²

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        ‖ u - uʳ ‖²_L²(Ω) = ∫_Ω |u(x) - uʳ(x)|² dx

    The channel axis is summed **after** the aggregation.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
        If not specified, the MSE is computed against zero, i.e., the norm of
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor.
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
    )


def nMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the normalized mean squared error (nMSE) between two states.

        ∑_(channels) [∑_(space) (L/N)ᴰ |uₕ - uₕʳ|² / ∑_(space) (L/N)ᴰ |uₕʳ|²]

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        ‖ u - uʳ ‖²_L²(Ω) / ‖ uʳ ‖²_L²(Ω) = ∫_Ω |u(x) - uʳ(x)|² dx / ∫_Ω |uʳ(x)|² dx

    The channel axis is summed **after** the aggregation.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor.
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
    )


def sMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the symmetric mean squared error (sMSE) between two states.

        ∑_(channels) [2 ∑_(space) (L/N)ᴰ |uₕ - uₕʳ|² / (∑_(space) (L/N)ᴰ |uₕ|² + ∑_(space) (L/N)ᴰ |uₕʳ|²)]

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        2 ∫_Ω |u(x) - uʳ(x)|² dx / (∫_Ω |u(x)|² dx + ∫_Ω |uʳ(x)|² dx)

    The channel axis is summed **after** the aggregation.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! info
        This symmetric metric is bounded between 0 and C with C being the number
        of channels.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor.
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="symmetric",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=1.0,
    )


def RMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Optional[Float[Array, "C ... N"]] = None,
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the root mean squared error (RMSE) between two states.

        (∑_(channels) √(∑_(space) (L/N)ᴰ |uₕ - uₕʳ|²))

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        (‖ u - uʳ ‖_L²(Ω)) = √(∫_Ω |u(x) - uʳ(x)|² dx)

    The channel axis is summed **after** the aggregation. Hence, it is also
    summed **after** the square root. If you need the RMSE per channel, consider
    using [`exponax.metrics.spatial_aggregator`][] directly.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
        If not specified, the RMSE is computed against zero, i.e., the norm of
        `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="absolute",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=0.5,
    )


def nRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the normalized root mean squared error (nRMSE) between two states.

        ∑_(channels) [√(∑_(space) (L/N)ᴰ |uₕ - uₕʳ|²) / √(∑_(space) (L/N)ᴰ
        |uₕʳ|²)]

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        (‖ u - uʳ ‖_L²(Ω) / ‖ uʳ ‖_L²(Ω)) = √(∫_Ω |u(x) - uʳ(x)|² dx / ∫_Ω
        |uʳ(x)|² dx

    The channel axis is summed **after** the aggregation. Hence, it is also
    summed **after** the square root and after normalization. If you need more
    fine-grained control, consider using
    [`exponax.metrics.spatial_aggregator`][] directly.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only
        contributes a multiplicative factor
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="normalized",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=0.5,
    )


def sRMSE(
    u_pred: Float[Array, "C ... N"],
    u_ref: Float[Array, "C ... N"],
    *,
    domain_extent: float = 1.0,
) -> float:
    """
    Compute the symmetric root mean squared error (sRMSE) between two states.

        ∑_(channels) [2 √(∑_(space) (L/N)ᴰ |uₕ - uₕʳ|²) / (√(∑_(space) (L/N)ᴰ
        |uₕ|²) + √(∑_(space) (L/N)ᴰ |uₕʳ|²))]

    Given the correct `domain_extent`, this is consistent to the following
    functional norm:

        2 √(∫_Ω |u(x) - uʳ(x)|² dx) / (√(∫_Ω |u(x)|² dx) + √(∫_Ω |uʳ(x)|² dx))

    The channel axis is summed **after** the aggregation. Hence, it is also
    summed **after** the square root and after normalization. If you need more
    fine-grained control, consider using
    [`exponax.metrics.spatial_aggregator`][] directly.

    !!! tip
        To apply this function to a state tensor with a leading batch axis, use
        `jax.vmap`. Then the batch axis can be reduced, e.g., by `jnp.mean`. As
        a helper for this, [`exponax.metrics.mean_metric`][] is provided.

    !!! info
        This symmetric metric is bounded between 0 and C with C being the number
        of channels.


    **Arguments:**

    - `u_pred`: The state array, must follow the `Exponax` convention with a
        leading channel axis, and either one, two, or three subsequent spatial
        axes.
    - `u_ref`: The reference state array. Must have the same shape as `u_pred`.
    - `domain_extent`: The extent `L` of the domain `Ω = (0, L)ᴰ`. Must be
        provide to get the correctly consistent norm. If this metric is used an
        optimization objective, it can often be ignored since it only contributes
        a multiplicative factor
    """
    return spatial_norm(
        u_pred,
        u_ref,
        mode="symmetric",
        domain_extent=domain_extent,
        inner_exponent=2.0,
        outer_exponent=0.5,
    )
