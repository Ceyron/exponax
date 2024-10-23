import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._spectral import build_laplace_operator, build_scaling_array, build_wavenumbers
from ._base import BaseNonlinearFun


class VorticityConvection2d(BaseNonlinearFun):
    convection_scale: float
    derivative_operator: Complex[Array, "D ... (N//2)+1"]
    inv_laplacian: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        convection_scale: float = 1.0,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        """
        Performs a pseudo-spectral evaluation of the nonlinear vorticity
        convection, e.g., found in the 2d Navier-Stokes equations in
        streamfunction-vorticity formulation. In state space, it reads

        ```
            𝒩(u) = - b ([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u
        ```

        with `b` the convection scale, `⊙` the Hadamard product, `∇` the
        derivative operator, `Δ⁻¹` the inverse Laplacian, and `u` the vorticity.

        The minus arises because `Exponax` follows the convention that all
        nonlinear and linear differential operators are on the right-hand side
        of the equation. Typically, the vorticity convection term is on the
        left-hand side. Hence, the minus is required to move the term to the
        right-hand side.

        Since the inverse Laplacian is required, it internally performs a
        Poisson solve which is straightforward in Fourier space.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same.
        - `convection_scale`: The scale `b` of the convection term. Defaults
            to `1.0`.
        - `derivative_operator`: A complex array of shape `(d, ..., N//2+1)`
            that represents the derivative operator in Fourier space.
        - `dealiasing_fraction`: The fraction of the highest resolved modes
            that are not aliased. Defaults to `2/3` which corresponds to
            Orszag's 2/3 rule.
        """
        if num_spatial_dims != 2:
            raise ValueError(f"Expected num_spatial_dims = 2, got {num_spatial_dims}.")

        super().__init__(
            num_spatial_dims,
            num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

        self.convection_scale = convection_scale
        self.derivative_operator = derivative_operator

        laplacian = build_laplace_operator(derivative_operator, order=2)

        # Uses the UNCHANGED mean solution to the Poisson equation (hence, the
        # mean of the "right-hand side" will be the mean of the solution).
        # However, this does not matter because we subsequently take the
        # gradient which would annihilate any mean energy anyway.
        self.inv_laplacian = jnp.where(laplacian == 0, 1.0, 1 / laplacian)

    def __call__(
        self, u_hat: Complex[Array, "1 ... (N//2)+1"]
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        vorticity_hat = u_hat
        stream_function_hat = self.inv_laplacian * vorticity_hat

        u_hat = +self.derivative_operator[1:2] * stream_function_hat
        v_hat = -self.derivative_operator[0:1] * stream_function_hat
        del_vorticity_del_x_hat = self.derivative_operator[0:1] * vorticity_hat
        del_vorticity_del_y_hat = self.derivative_operator[1:2] * vorticity_hat

        u = self.ifft(self.dealias(u_hat))
        v = self.ifft(self.dealias(v_hat))
        del_vorticity_del_x = self.ifft(self.dealias(del_vorticity_del_x_hat))
        del_vorticity_del_y = self.ifft(self.dealias(del_vorticity_del_y_hat))

        convection = u * del_vorticity_del_x + v * del_vorticity_del_y

        convection_hat = self.fft(convection)

        # Need minus to bring term to the right-hand side
        return -self.convection_scale * convection_hat


class VorticityConvection2dKolmogorov(VorticityConvection2d):
    injection: Complex[Array, "1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        convection_scale: float = 1.0,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        """
        Performs a pseudo-spectral evaluation of the nonlinear vorticity
        convection together with a Kolmorogov-like injection term, e.g., found
        in the 2d Navier-Stokes equations in streamfunction-vorticity
        formulation used for simulating a Kolmogorov flow.

        In state space, it reads

        ```
            𝒩(u) = - b ([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u - f
        ```

        For details on the vorticity convective term, see
        `exponax.nonlin_fun.VorticityConvection2d`. The forcing term has the
        form

        ```
            f = -k (2π/L) γ cos(k (2π/L) x₁)
        ```

        i.e., energy of intensity `γ` is injected at wavenumber `k`. Note that
        the forcing is on the **vorticity**. As such, we get the prefactor `k
        (2π/L)` and the `sin(...)` turns into a `-cos(...)` (minus sign because
        the vorticity is derived via the curl).

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same.
        - `convection_scale`: The scale `b` of the convection term. Defaults
            to `1.0`.
        - `injection_mode`: The wavenumber `k` at which energy is injected.
        - `injection_scale`: The intensity `γ` of the injection term.
        - `derivative_operator`: A complex array of shape `(d, ..., N//2+1)`
            that represents the derivative operator in Fourier space.
        - `dealiasing_fraction`: The fraction of the highest resolved modes
            that are not aliased. Defaults to `2/3` which corresponds to
            Orszag's 2/3 rule.
        """
        super().__init__(
            num_spatial_dims,
            num_points,
            convection_scale=convection_scale,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

        wavenumbers = build_wavenumbers(num_spatial_dims, num_points)
        injection_mask = (wavenumbers[0] == 0) & (wavenumbers[1] == injection_mode)
        self.injection = jnp.where(
            injection_mask,
            # Need to additional scale the `injection_scale` with the
            # `injection_mode`, because we apply the forcing on the vorticity.
            -injection_mode
            * injection_scale
            * build_scaling_array(num_spatial_dims, num_points, mode="coef_extraction"),
            0.0,
        )

    def __call__(
        self, u_hat: Complex[Array, "1 ... (N//2)+1"]
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        neg_convection_hat = super().__call__(u_hat)
        return neg_convection_hat + self.injection
