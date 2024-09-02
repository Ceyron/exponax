import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._base_ic import BaseIC, BaseRandomICGenerator


class MultiChannelIC(eqx.Module):
    initial_conditions: tuple[BaseIC, ...]

    def __init__(self, initial_conditions: tuple[BaseIC, ...]):
        """
        A multi-channel initial condition.

        **Arguments**:

        - `initial_conditions`: A tuple of initial conditions.
        """
        self.initial_conditions = initial_conditions

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

    def __init__(self, ic_generators: tuple[BaseRandomICGenerator, ...]):
        """
        A multi-channel random initial condition generator. Use this for
        problems with multiple channels, like Burgers in higher dimensions or
        the Gray-Scott dynamics.

        **Arguments**:

        - `ic_generators`: A tuple of initial condition generators.

        !!! example
            Below is an example for generating a random multi-channel initial
            condition for the three-dimensional Burgers equation which has three
            channels. For simplicity, we will use the same IC generator for each
            channel.

            ```python
            import jax
            import exponax as ex

            single_channel_ic_gen = ex.ic.RandomTruncatedFourierSeries(
                3,
                max_one=True,
            )
            multi_channel_ic_gen = ex.ic.RandomMultiChannelICGenerator(
                [single_channel_ic_gen,] * 3
            )

            ic = multi_channel_ic_gen(100, key=jax.random.PRNGKey(0))
            ```
        """
        self.ic_generators = ic_generators

    def gen_ic_fun(self, *, key: PRNGKeyArray) -> MultiChannelIC:
        ic_funs = [
            ic_gen.gen_ic_fun(key=k)
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
