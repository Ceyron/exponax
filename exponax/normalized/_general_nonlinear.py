import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import GeneralNonlinearFun1d


class NormlizedGeneralNonlinearStepper1d(BaseStepper):
    normalized_coefficients_linear: list[float]
    normalized_coefficients_nonlinear: list[float]
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_coefficients_linear: list[float] = [0.0, 0.0, 0.01 * 0.1],
        normalized_coefficients_nonlinear: list[float] = [0.0, -1.0 * 0.1, 0.0],
        order=2,
        dealiasing_fraction: float = 2 / 3,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        By default Burgers.
        """

        if num_spatial_dims != 1:
            raise ValueError(
                "The number of spatial dimensions must be 1 because of ambiguity in channel growth"
            )
        if len(normalized_coefficients_nonlinear) != 3:
            raise ValueError(
                "The nonlinear coefficients list must have exactly 3 elements"
            )
        self.normalized_coefficients_linear = normalized_coefficients_linear
        self.normalized_coefficients_nonlinear = normalized_coefficients_nonlinear
        self.dealiasing_fraction = dealiasing_fraction

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,  # Derivative operator is just scaled with 2 * jnp.pi
            num_points=num_points,
            dt=1.0,
            num_channels=1,
            order=order,
            n_circle_points=n_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(self, derivative_operator: Array) -> Array:
        linear_operator = sum(
            jnp.sum(
                c * (derivative_operator) ** i,
                axis=0,
                keepdims=True,
            )
            for i, c in enumerate(self.normalized_coefficients_linear)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> GeneralNonlinearFun1d:
        return GeneralNonlinearFun1d(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale_list=self.normalized_coefficients_nonlinear,
            zero_mode_fix=True,  # ToDo: check this
        )