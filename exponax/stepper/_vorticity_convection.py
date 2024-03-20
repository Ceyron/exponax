import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from ..nonlin_fun import VorticityConvection2d, VorticityConvection2dKolmogorov


class GeneralVorticityConvectionStepper(BaseStepper):
    vorticity_convection_scale: float
    coefficients: tuple[float, ...]
    injection_mode: int
    injection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        vorticity_convection_scale: float = 1.0,
        coefficients: tuple[float, ...] = (0.0, 0.0, 0.001),
        injection_mode: int = 4,
        injection_scale: float = 0.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        if num_spatial_dims != 2:
            raise ValueError(f"Expected num_spatial_dims = 2, got {num_spatial_dims}.")
        self.vorticity_convection_scale = vorticity_convection_scale
        self.coefficients = coefficients
        self.injection_mode = injection_mode
        self.injection_scale = injection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
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
            for i, c in enumerate(self.coefficients)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> VorticityConvection2d:
        if self.injection_scale == 0.0:
            return VorticityConvection2d(
                self.num_spatial_dims,
                self.num_points,
                convection_scale=self.vorticity_convection_scale,
                derivative_operator=derivative_operator,
                dealiasing_fraction=self.dealiasing_fraction,
            )
        else:
            return VorticityConvection2dKolmogorov(
                self.num_spatial_dims,
                self.num_points,
                convection_scale=self.vorticity_convection_scale,
                derivative_operator=derivative_operator,
                dealiasing_fraction=self.dealiasing_fraction,
                injection_mode=self.injection_mode,
                injection_scale=self.injection_scale,
            )
