from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun
from ._gradient_norm import GradientNormNonlinearFun
from ._polynomial import PolynomialNonlinearFun
from ._single_channel_convection import SingleChannelConvectionNonlinearFun


class GeneralNonlinearFun(BaseNonlinearFun):
    square_nonlinear_fun: PolynomialNonlinearFun
    convection_nonlinear_fun: SingleChannelConvectionNonlinearFun
    gradient_norm_nonlinear_fun: GradientNormNonlinearFun

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        scale_list: list[float] = [0.0, -1.0, 0.0],
        zero_mode_fix: bool = True,
    ):
        """
        Uses an additional scaling of 0.5 on the latter two components only

        By default: Burgers equation
        """
        if len(scale_list) != 3:
            raise ValueError("The scale list must have exactly 3 elements")

        self.square_nonlinear_fun = PolynomialNonlinearFun(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
            coefficients=[0.0, 0.0, scale_list[0]],
        )
        self.convection_nonlinear_fun = SingleChannelConvectionNonlinearFun(
            num_spatial_dims,
            num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
            # Minus required because it internally has another minus
            scale=-scale_list[1],
        )
        self.gradient_norm_nonlinear_fun = GradientNormNonlinearFun(
            num_spatial_dims,
            num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
            # Minus required because it internally has another minus
            scale=-scale_list[2],
            zero_mode_fix=zero_mode_fix,
        )

        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        return (
            self.square_nonlinear_fun(u_hat)
            + self.convection_nonlinear_fun(u_hat)
            + self.gradient_norm_nonlinear_fun(u_hat)
        )
