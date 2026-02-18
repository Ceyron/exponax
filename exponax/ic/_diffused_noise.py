from jaxtyping import Array, Float, PRNGKeyArray

from ..stepper import Diffusion
from ._base_ic import (
    BaseRandomICGenerator,
    normalize_ic,
    validate_normalization_options,
)
from ._white_noise import WhiteNoise


class DiffusedNoise(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    intensity: float
    zero_mean: bool
    std_one: bool
    max_one: bool
    white_noise: WhiteNoise

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        domain_extent: float = 1.0,
        intensity=0.001,
        zero_mean: bool = True,
        std_one: bool = False,
        max_one: bool = False,
    ):
        """
        Randomly generated initial condition consisting of a diffused noise
        field.

        The original noise is drawn in state space with a uniform normal
        distribution. After the application of the diffusion operator, the
        spectrum decays exponentially quadratic with a rate of `intensity`.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The extent of the domain. Defaults to `1.0`. This
            indirectly affects the intensity of the noise. It is best to keep it
            at `1.0` and just adjust the `intensity` instead.
        - `intensity`: The intensity of the noise. Defaults to `0.001`.
        - `zero_mean`: Whether to zero the mean of the noise.
        - `std_one`: Whether to normalize the noise to have a standard
            deviation of one. Defaults to `False`.
        - `max_one`: Whether to normalize the noise to the maximum absolute
            value of one. Defaults to `False`.
        """
        validate_normalization_options(
            zero_mean=zero_mean, std_one=std_one, max_one=max_one
        )
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.intensity = intensity
        self.zero_mean = zero_mean
        self.std_one = std_one
        self.max_one = max_one
        self.white_noise = WhiteNoise(num_spatial_dims)

    def __call__(
        self, num_points: int, *, key: PRNGKeyArray
    ) -> Float[Array, "1 ... N"]:
        noise = self.white_noise(num_points, key=key)

        diffusion_stepper = Diffusion(
            self.num_spatial_dims,
            self.domain_extent,
            num_points,
            1.0,
            diffusivity=self.intensity,
        )
        ic = diffusion_stepper(noise)

        ic = normalize_ic(
            ic, zero_mean=self.zero_mean, std_one=self.std_one, max_one=self.max_one
        )

        return ic
