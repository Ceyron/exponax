from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun
from ._convection import ConvectionNonlinearFun
from ._gradient_norm import GradientNormNonlinearFun
from ._polynomial import PolynomialNonlinearFun


class GeneralNonlinearFun(BaseNonlinearFun):
    square_nonlinear_fun: PolynomialNonlinearFun
    convection_nonlinear_fun: ConvectionNonlinearFun
    gradient_norm_nonlinear_fun: GradientNormNonlinearFun

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        scale_list: tuple[float, float, float] = (0.0, -1.0, 0.0),
        zero_mode_fix: bool = True,
    ):
        """
        Fourier pseudo-spectral evaluation of a nonlinear differential operator
        that has a square, convection (with single-channel hack), and gradient
        norm term. In 1D and state space, this reads

        ```
            𝒩(u) = b₀ u² + b₁ 1/2 (u²)ₓ + b₂ 1/2 (uₓ)²
        ```

        The higher-dimensional extension is designed for a single-channel state
        `u` (i.e., the number of channels do not grow with the number of spatial
        dimensions, see also the description of
        `exponax.nonlin_fun.ConvectionNonlinearFun`). The extension reads

        ```
            𝒩(u) = b₀ u² + b₁ 1/2 (1⃗ ⋅ ∇)(u²) + b₂ 1/2 ‖∇u‖₂²
        ```

        !!! warning
            In contrast to the individual nonlinear functions
            `exponax.nonlin_fun.ConvectionNonlinearFun` and
            `exponax.nonlin_fun.GradientNormNonlinearFun`, there is no minus.
            Hence, to have a "propoper" convection term, consider supplying a
            negative scale for the convection term, etc.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions `D`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same.
        - `derivative_operator`: A complex array of shape `(D, ..., N//2+1)`
            that represents the derivative operator in Fourier space.
        - `dealiasing_fraction`: The fraction of the highest resolved modes that
            are not aliased. Defaults to `2/3` which corresponds to Orszag's 2/3
            rule.
        - `scale_list`: A tuple of three floats `[b₀, b₁, b₂]` that represent
            the scales of the square, (single-channel) convection, and gradient
            norm term, respectively. Defaults to `[0.0, -1.0, 0.0]` which
            corresponds to a pure convection term (i.e, in 1D together with a
            diffusion linear term, this would be the Burgers equation). !!!
            important: note that negation has to be manually provided!
        - `zero_mode_fix`: Whether to set the zero mode to zero. In other words,
            whether to have mean zero energy after nonlinear function activation.
            This exists because the nonlinear operation happens after the
            derivative operator is applied. Naturally, the derivative sets any
            constant offset to zero. However, the square nonlinearity introduces
            again a new constant offset. Setting this argument to `True` removes
            this offset. Defaults to `True`.
        """
        if len(scale_list) != 3:
            raise ValueError("The scale list must have exactly 3 elements")

        self.square_nonlinear_fun = PolynomialNonlinearFun(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
            coefficients=[0.0, 0.0, scale_list[0]],
        )
        self.convection_nonlinear_fun = ConvectionNonlinearFun(
            num_spatial_dims,
            num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
            # Minus required because it internally has another minus
            scale=-scale_list[1],
            single_channel=True,
            # For legacy reasons, the single-channel convection term is conservative
            conservative=True,
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
