import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._base_ic import BaseIC, BaseRandomICGenerator


class MultiChannelIC(eqx.Module):
    initial_conditions: tuple[BaseIC, ...]

    def __call__(self, x: Float[Array, "D ... N"]) -> Float[Array, "C ... N"]:
        """
        Evaluate the initial condition.

        **Arguments**:
            - `x`: The grid points.

        **Returns**:
            - `u`: The initial condition evaluated at the grid points.
        """
        return jnp.concatenate([ic(x) for ic in self.initial_conditions], axis=0)


class RandomMultiChannelICGenerator(eqx.Module):
    ic_generators: tuple[BaseRandomICGenerator, ...]

    def gen_ic_fun(self, num_points: int, *, key: PRNGKeyArray) -> MultiChannelIC:
        ic_funs = [
            ic_gen.gen_ic_fun(num_points, key=k)
            for (ic_gen, k) in zip(
                self.ic_generators,
                jax.random.split(key, len(self.ic_generators)),
            )
        ]
        return MultiChannelIC(ic_funs)

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "C ... N"]:
        u_list = [
            ic_gen(num_points, key=k)
            for (ic_gen, k) in zip(
                self.ic_generators,
                jax.random.split(key, len(self.ic_generators)),
            )
        ]
        return jnp.concatenate(u_list, axis=0)
