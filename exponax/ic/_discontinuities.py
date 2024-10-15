import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from ._base_ic import BaseIC, BaseRandomICGenerator


class Discontinuity(eqx.Module):
    """
    A state described by a discontinuity with a constant value within the
    hypercube defined by the lower and upper limits.
    """

    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    value: float

    def __call__(self, x: Array) -> Array:
        mask = jnp.ones_like(x, dtype=bool)
        for i, (lb, ub) in enumerate(zip(self.lower_limits, self.upper_limits)):
            mask = mask & (x[i : i + 1] > lb) & (x[i : i + 1] < ub)

        return jnp.where(mask, self.value, 0.0)


class Discontinuities(BaseIC):
    discontinuity_list: tuple[Discontinuity, ...]
    zero_mean: bool
    std_one: bool
    max_one: bool

    def __init__(
        self,
        discontinuity_list: tuple[Discontinuity, ...],
        *,
        zero_mean: bool = True,
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        A state described by a collection of discontinuities.

        **Arguments**:

        - `discontinuity_list`: A tuple of discontinuities.
        - `zero_mean`: Whether the state should have zero mean.
        - `std_one`: Whether to normalize the state to have a standard
            deviation of one. Defaults to `False`. Only works if the offset is
            zero.
        - `max_one`: Whether to normalize the state to have the maximum
            absolute value of one. Defaults to `False`. Only one of `std_one`
            and `max_one` can be `True`.
        """
        if not zero_mean and std_one:
            raise ValueError("Cannot have `zero_mean=False` and `std_one=True`.")
        if std_one and max_one:
            raise ValueError("Cannot have `std_one=True` and `max_one=True`.")

        self.discontinuity_list = discontinuity_list
        self.zero_mean = zero_mean
        self.std_one = std_one
        self.max_one = max_one

    def __call__(self, x: Array) -> Array:
        ic = sum(disc(x) for disc in self.discontinuity_list)

        if self.zero_mean:
            ic = ic - jnp.mean(ic)

        if self.std_one:
            ic = ic / jnp.std(ic)

        if self.max_one:
            ic = ic / jnp.max(jnp.abs(ic))

        return ic


class RandomDiscontinuities(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    num_discontinuities: int
    value_range: tuple[float, float]

    zero_mean: bool
    std_one: bool
    max_one: bool

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        domain_extent: float = 1.0,
        num_discontinuities: int = 3,
        value_range: tuple[float, float] = (-1.0, 1.0),
        zero_mean: bool = False,
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        Random generator for initial states described by a collection of
        discontinuities.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions.
        - `domain_extent`: The extent of the domain in each spatial direction.
        - `num_discontinuities`: The number of discontinuities.
        - `value_range`: The range of values for the discontinuities.
        - `zero_mean`: Whether the state should have zero mean.
        - `std_one`: Whether to normalize the state to have a standard
            deviation of one. Defaults to `False`. Only works if the offset is
            zero.
        - `max_one`: Whether to normalize the state to have the maximum
            absolute value of one. Defaults to `False`. Only one of `std_one`
            and `max_one` can be `True`.
        """
        if not zero_mean and std_one:
            raise ValueError("Cannot have `zero_mean=False` and `std_one=True`.")
        if std_one and max_one:
            raise ValueError("Cannot have `std_one=True` and `max_one=True`.")

        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_discontinuities = num_discontinuities
        self.value_range = value_range

        self.zero_mean = zero_mean
        self.std_one = std_one
        self.max_one = max_one

    def gen_one_ic_fn(self, *, key: PRNGKeyArray) -> Discontinuity:
        """
        Generates a single discontinuity.
        """
        lower_limits = []
        upper_limits = []
        for i in range(self.num_spatial_dims):
            key_1, key_2, key = jr.split(key, 3)
            lim_1 = jr.uniform(key_1, (), minval=0.0, maxval=self.domain_extent)
            lim_2 = jr.uniform(key_2, (), minval=0.0, maxval=self.domain_extent)
            lower_limits.append(jnp.minimum(lim_1, lim_2))
            upper_limits.append(jnp.maximum(lim_1, lim_2))

        lower_limits = tuple(lower_limits)
        upper_limits = tuple(upper_limits)

        value = jr.uniform(
            key, (), minval=self.value_range[0], maxval=self.value_range[1]
        )

        return Discontinuity(
            lower_limits=lower_limits, upper_limits=upper_limits, value=value
        )

    def gen_ic_fun(self, *, key: PRNGKeyArray) -> BaseIC:
        disc_list = [
            self.gen_one_ic_fn(key=k) for k in jr.split(key, self.num_discontinuities)
        ]
        return Discontinuities(
            discontinuity_list=disc_list,
            zero_mean=self.zero_mean,
            std_one=self.std_one,
            max_one=self.max_one,
        )
