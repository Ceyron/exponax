import jax.numpy as jnp
from jaxtyping import Array, Complex

from ._base import BaseNonlinearFun


class ConvectionNonlinearFun(BaseNonlinearFun):
    derivative_operator: Complex[Array, "D ... (N//2)+1"]
    scale: float
    single_channel: bool

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float = 2 / 3,
        scale: float = 1.0,
        single_channel: bool = False,
    ):
        """
        Performs a pseudo-spectral evaluation of the nonlinear convection, e.g.,
        found in the Burgers equation. In 1d and state space, this reads

        ```
            𝒩(u) = b₁ 1/2 (u²)ₓ
        ```

        with a scale `b₁`. The typical extension to higher dimensions requires u
        to have as many channels as spatial dimensions and then gives

        ```
            𝒩(u) = b₁ 1/2 ∇ ⋅ (u ⊗ u)
        ```

        with `∇ ⋅` the divergence operator and the outer product `u ⊗ u`.
        Another option is a "single-channel" hack requiring only one channel no
        matter the spatial dimensions. This reads

        ```
            𝒩(u) = b₁ 1/2 (1⃗ ⋅ ∇)(u²)
        ```

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
        - `scale`: The scale `b₁` of the convection term. Defaults to `1.0`.
        - `single_channel`: Whether to use the single-channel hack. Defaults
            to `False`.
        """
        self.derivative_operator = derivative_operator
        self.scale = scale
        self.single_channel = single_channel
        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

    def _multi_channel_eval(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Evaluates the convection term for a multi-channel state `u_hat` in
        Fourier space. The convection term is given by

        ```
            𝒩(u) = b₁ 1/2 ∇ ⋅ (u ⊗ u)
        ```

        with `∇ ⋅` the divergence operator and the outer product `u ⊗ u`.

        **Arguments:**

        - `u_hat`: The state in Fourier space.

        **Returns:**

        - `convection`: The evaluation of the convection term in Fourier space.
        """
        num_channels = u_hat.shape[0]
        if num_channels != self.num_spatial_dims:
            raise ValueError(
                "Number of channels in u_hat should match number of spatial dimensions"
            )
        u_hat_dealiased = self.dealias(u_hat)
        u = self.ifft(u_hat_dealiased)
        u_outer_product = u[:, None] * u[None, :]
        u_outer_product_hat = self.fft(u_outer_product)
        convection = 0.5 * jnp.sum(
            self.derivative_operator[None, :] * u_outer_product_hat,
            axis=1,
        )
        # Requires minus to move term to the rhs
        return -self.scale * convection

    def _single_channel_eval(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Evaluates the convection term for a single-channel state `u_hat` in
        Fourier space. The convection term is given by

        ```
            𝒩(u) = b₁ 1/2 (1⃗ ⋅ ∇)(u²)
        ```

        with `∇ ⋅` the divergence operator and `1⃗` a vector of ones.

        **Arguments:**

        - `u_hat`: The state in Fourier space.

        **Returns:**

        - `convection`: The evaluation of the convection term in Fourier space.
        """
        u_hat_dealiased = self.dealias(u_hat)
        u = self.ifft(u_hat_dealiased)
        u_square = u**2
        u_square_hat = self.fft(u_square)
        sum_of_derivatives_operator = jnp.sum(
            self.derivative_operator, axis=0, keepdims=True
        )
        convection = 0.5 * sum_of_derivatives_operator * u_square_hat
        # Requires minus to bring convection to the right-hand side
        return -self.scale * convection

    def __call__(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        if self.single_channel:
            return self._single_channel_eval(u_hat)
        else:
            return self._multi_channel_eval(u_hat)
