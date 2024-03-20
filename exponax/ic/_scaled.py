from jaxtyping import Array, Float, PRNGKeyArray

from ._base_ic import BaseIC, BaseRandomICGenerator


class ScaledIC(BaseIC):
    ic: BaseIC
    scale: float

    def __call__(self, x: Float[Array, "D ... N"]) -> Float[Array, "1 ... N"]:
        return self.ic(x) * self.scale


class ScaledICGenerator(BaseRandomICGenerator):
    """
    Works best in combination with initial conditions that have `max_one=True`
    or `std_one=True`.
    """

    ic_gen: BaseRandomICGenerator
    scale: float

    def gen_ic_fun(self, *, key: PRNGKeyArray) -> BaseIC:
        return ScaledIC(self.ic_gen.gen_ic_fun(key=key), scale=self.scale)

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        ic = self.ic_gen(num_points=num_points, key=key)
        return ic * self.scale
