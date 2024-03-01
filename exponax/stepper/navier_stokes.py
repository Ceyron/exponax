from jaxtyping import Array, Complex

from ..base_stepper import BaseStepper
from ..nonlin_fun import VorticityConvection2d, VorticityConvection2dKolmogorov
from ..spectral import build_laplace_operator


class NavierStokesVorticity2d(BaseStepper):
    diffusivity: float
    drag: float
    dealiasing_fraction: float

    def __init__(
        self,
        # Does not require D argument as it is fixed to 2
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
        drag: float = 0.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.diffusivity = diffusivity
        self.drag = drag
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=2,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
            n_circle_points=n_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        return self.diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) + self.drag * build_laplace_operator(derivative_operator, order=0)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> VorticityConvection2d:
        return VorticityConvection2d(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
        )


class KolmogorovFlowVorticity2d(BaseStepper):
    diffusivity: float
    drag: float
    injection_mode: int
    injection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        # Does not require D argument as it is fixed to 2
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.001,
        drag: float = -0.1,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        n_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.diffusivity = diffusivity
        self.drag = drag
        self.injection_mode = injection_mode
        self.injection_scale = injection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=2,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
            n_circle_points=n_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        return self.diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) + self.drag * build_laplace_operator(derivative_operator, order=0)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> VorticityConvection2dKolmogorov:
        return VorticityConvection2dKolmogorov(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            num_channels=self.num_channels,
            injection_mode=self.injection_mode,
            injection_scale=self.injection_scale,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
        )
