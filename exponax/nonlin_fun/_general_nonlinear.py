from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun
from ._convection import ConvectionNonlinearFun
from ._gradient_norm import GradientNormNonlinearFun
from ._polynomial import PolynomialNonlinearFun


class GeneralNonlinearFun1d(BaseNonlinearFun):
    square_nonlinear_fun: PolynomialNonlinearFun
    convection_nonlinear_fun: ConvectionNonlinearFun
    gradient_norm_nonlinear_fun: GradientNormNonlinearFun

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        scale_list: list[float] = [0.0, -1.0, 0.0],
        zero_mode_fix: bool = False,
    ):
        """
        Uses an additional scaling of 0.5 on the latter two components only

        By default: Burgers equation
        """
        if num_spatial_dims != 1:
            raise ValueError("The number of spatial dimensions must be 1")
        if len(scale_list) != 3:
            raise ValueError("The scale list must have exactly 3 elements")

        self.square_nonlinear_fun = PolynomialNonlinearFun(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            num_channels=num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
            coefficients=[0.0, 0.0, scale_list[0]],
        )
        self.convection_nonlinear_fun = ConvectionNonlinearFun(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            num_channels=num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
            # Minus required because it internally has another minus
            scale=-scale_list[1],
            zero_mode_fix=zero_mode_fix,
        )
        self.gradient_norm_nonlinear_fun = GradientNormNonlinearFun(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            num_channels=num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
            # Minus required because it internally has another minus
            scale=-scale_list[2],
        )

        super().__init__(
            num_spatial_dims,
            num_points,
            num_channels,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

    def evaluate(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return (
            self.square_nonlinear_fun.evaluate(u_hat)
            + self.convection_nonlinear_fun.evaluate(u_hat)
            + self.gradient_norm_nonlinear_fun.evaluate(u_hat)
        )
