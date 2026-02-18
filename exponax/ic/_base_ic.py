from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .._utils import make_grid


def validate_normalization_options(*, zero_mean: bool, std_one: bool, max_one: bool):
    """Raises ValueError for invalid normalization option combinations."""
    if not zero_mean and std_one:
        raise ValueError("Cannot have `zero_mean=False` and `std_one=True`.")
    if std_one and max_one:
        raise ValueError("Cannot have `std_one=True` and `max_one=True`.")


def normalize_ic(
    ic: Float[Array, "1 ... N"],
    *,
    zero_mean: bool = True,
    std_one: bool = False,
    max_one: bool = False,
) -> Float[Array, "1 ... N"]:
    """Apply normalization to an initial condition array."""
    if zero_mean:
        ic = ic - jnp.mean(ic)
    if std_one:
        ic = ic / jnp.std(ic)
    if max_one:
        ic = ic / jnp.max(jnp.abs(ic))
    return ic


class BaseIC(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, x: Float[Array, "D ... N"]) -> Float[Array, "1 ... N"]:
        """
        Evaluate the initial condition.

        **Arguments**:

        - `x`: The grid points.

        **Returns**:

        - `u`: The initial condition evaluated at the grid points.
        """
        pass


class BaseRandomICGenerator(eqx.Module):
    num_spatial_dims: int
    indexing: str = "ij"

    def gen_ic_fun(self, *, key: PRNGKeyArray) -> BaseIC:
        """
        Generate an initial condition function.

        **Arguments**:

        - `key`: A jax random key.

        **Returns**:

        - `ic`: An initial condition function that can be evaluated at
            degree of freedom locations.
        """
        raise NotImplementedError(
            """This random ic generator cannot represent its initial condition
            as a function. Directly evaluate it."""
        )

    def __call__(
        self,
        num_points: int,
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "1 ... N"]:
        """
        Generate a random initial condition on a grid with `num_points` points.

        **Arguments**:

        - `num_points`: The number of grid points in each dimension.
        - `key`: A jax random key.

        **Returns**:

        - `u`: The initial condition evaluated at the grid points.
        """
        ic_fun = self.gen_ic_fun(key=key)
        grid = make_grid(
            self.num_spatial_dims,
            self.domain_extent,
            num_points,
            indexing=self.indexing,
        )
        return ic_fun(grid)
