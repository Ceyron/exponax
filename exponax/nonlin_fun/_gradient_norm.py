import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._base import BaseNonlinearFun


class GradientNormNonlinearFun(BaseNonlinearFun):
    scale: float
    zero_mode_fix: bool
    derivative_operator: Complex[Array, "D ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
        zero_mode_fix: bool = True,
        scale: float = 1.0,
    ):
        """
        Performs a pseudo-spectral evaluation of the nonlinear gradient norm,
        e.g., found in the Kuramoto-Sivashinsky equation in combustion format.
        In 1d and state space, this reads

        ```
            𝒩(u) = - b₂ 1/2 (uₓ)²
        ```

        with a scale `b₂`. The minus arises because `Exponax` follows the
        convention that all nonlinear and linear differential operators are on
        the right-hand side of the equation. Typically, the gradient norm term
        is on the left-hand side. Hence, the minus is required to move the term
        to the right-hand side.

        In higher dimensions, u has to be single channel and the nonlinear
        function reads

        ```
            𝒩(u) = - b₂ 1/2 ‖∇u‖₂²
        ```

        with `‖∇u‖₂²` the squared L2 norm of the gradient of `u`.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same.
        - `derivative_operator`: A complex array of shape `(d, ..., N//2+1)`
            that represents the derivative operator in Fourier space.
        - `dealiasing_fraction`: The fraction of the highest resolved modes
            that are not aliased. Defaults to `2/3` which corresponds to
            Orszag's 2/3 rule.
        - `zero_mode_fix`: Whether to set the zero mode to zero. In other
            words, whether to have mean zero energy after nonlinear function
            activation. This exists because the nonlinear operation happens
            after the derivative operator is applied. Naturally, the derivative
            sets any constant offset to zero. However, the square nonlinearity
            introduces again a new constant offset. Setting this argument to
            `True` removes this offset. Defaults to `True`.
        - `scale`: The scale `b₂` of the gradient norm term. Defaults to
            `1.0`.
        """
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )
        self.derivative_operator = derivative_operator
        self.zero_mode_fix = zero_mode_fix
        self.scale = scale

    def zero_fix(
        self,
        f: Float[Array, "... N"],
    ):
        return f - jnp.mean(f)

    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        u_gradient_hat = self.derivative_operator[None, :] * u_hat[:, None]
        u_gradient = self.ifft(self.dealias(u_gradient_hat))

        # Reduces the axis introduced by the gradient
        u_gradient_norm_squared = jnp.sum(u_gradient**2, axis=1)

        if self.zero_mode_fix:
            # Maybe there is more efficient way
            u_gradient_norm_squared = jax.vmap(self.zero_fix)(u_gradient_norm_squared)

        u_gradient_norm_squared_hat = 0.5 * self.fft(u_gradient_norm_squared)

        # Requires minus to move term to the rhs
        return -self.scale * u_gradient_norm_squared_hat
