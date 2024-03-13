from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import ConvectionNonlinearFun, GradientNormNonlinearFun


class Nikolaevskiy(BaseStepper):
    gradient_norm_scale: float
    second_order_diffusivity: float
    fourth_order_diffusivity: float
    sixth_order_diffusivity: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        gradient_norm_scale: float = 1.0,
        second_order_diffusivity: float = 0.1,
        fourth_order_diffusivity: float = 1.0,
        sixth_order_diffusivity: float = 1.0,
        dealiasing_fraction: float = 2 / 3,
        order: int = 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.gradient_norm_scale = gradient_norm_scale
        self.second_order_diffusivity = second_order_diffusivity
        self.fourth_order_diffusivity = fourth_order_diffusivity
        self.sixth_order_diffusivity = sixth_order_diffusivity
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
        linear_operator = (
            self.second_order_diffusivity
            * build_laplace_operator(derivative_operator, order=2)
            + self.fourth_order_diffusivity
            * build_laplace_operator(derivative_operator, order=4)
            + self.sixth_order_diffusivity
            * build_laplace_operator(derivative_operator, order=6)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> GradientNormNonlinearFun:
        return GradientNormNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            zero_mode_fix=True,
            scale=self.gradient_norm_scale,
        )


class NikolaevskiyConservative(BaseStepper):
    convection_scale: float
    second_order_diffusivity: float
    fourth_order_diffusivity: float
    sixth_order_diffusivity: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        convection_scale: float = 1.0,
        second_order_diffusivity: float = 0.1,
        fourth_order_diffusivity: float = 1.0,
        sixth_order_diffusivity: float = 1.0,
        dealiasing_fraction: float = 2 / 3,
        order: int = 2,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.convection_scale = convection_scale
        self.second_order_diffusivity = second_order_diffusivity
        self.fourth_order_diffusivity = fourth_order_diffusivity
        self.sixth_order_diffusivity = sixth_order_diffusivity
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=num_spatial_dims,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        linear_operator = (
            self.second_order_diffusivity
            * build_laplace_operator(derivative_operator, order=2)
            + self.fourth_order_diffusivity
            * build_laplace_operator(derivative_operator, order=4)
            + self.sixth_order_diffusivity
            * build_laplace_operator(derivative_operator, order=6)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ConvectionNonlinearFun:
        return ConvectionNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
            scale=self.convection_scale,
        )
