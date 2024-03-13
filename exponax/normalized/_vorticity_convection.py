import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import VorticityConvection2d, VorticityConvection2dKolmogorov


class NormalizedVorticityConvection(BaseStepper):
    normalized_vorticity_convection_scale: float
    normalized_coefficients: tuple[float, ...]
    injection_mode: int
    normalized_injection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        normalized_vorticity_convection_scale: float = 0.01 * 1.0 / (1.0**0),
        normalized_coefficients: tuple[float, ...] = (
            0.01 * 0.0 / (1.0**0),
            0.0,
            0.01 * 0.001 / (1.0**2),
        ),
        injection_mode: int = 4,
        normalized_injection_scale: float = 0.01 * 0.0 / (1.0**0),
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        if num_spatial_dims != 2:
            raise ValueError(f"Expected num_spatial_dims = 2, got {num_spatial_dims}.")
        self.normalized_vorticity_convection_scale = (
            normalized_vorticity_convection_scale
        )
        self.normalized_coefficients = normalized_coefficients
        self.injection_mode = injection_mode
        self.normalized_injection_scale = normalized_injection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,
            num_points=num_points,
            dt=1.0,
            num_channels=1,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        linear_operator = sum(
            jnp.sum(
                c * (derivative_operator) ** i,
                axis=0,
                keepdims=True,
            )
            for i, c in enumerate(self.normalized_coefficients)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> VorticityConvection2d:
        if self.normalized_injection_scale == 0.0:
            return VorticityConvection2d(
                self.num_spatial_dims,
                self.num_points,
                convection_scale=self.normalized_vorticity_convection_scale,
                derivative_operator=derivative_operator,
                dealiasing_fraction=self.dealiasing_fraction,
            )
        else:
            return VorticityConvection2dKolmogorov(
                self.num_spatial_dims,
                self.num_points,
                convection_scale=self.normalized_vorticity_convection_scale,
                derivative_operator=derivative_operator,
                dealiasing_fraction=self.dealiasing_fraction,
                injection_mode=self.injection_mode,
                injection_scale=self.normalized_injection_scale,
            )
