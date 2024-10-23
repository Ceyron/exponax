from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from .._utils import make_grid


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
            "This random ic generator cannot represent its initial condition as a function. Directly evaluate it."
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
