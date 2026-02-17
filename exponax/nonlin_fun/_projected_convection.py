import jax.numpy as jnp
from jaxtyping import Array, Complex, Inexact

from .._spectral import build_scaling_array, build_wavenumbers
from ._base import BaseNonlinearFun
from ._leray import Leray


def _cross_product_3d(
    a: Inexact[Array, " 3 N N (N//2+1) "],
    b: Inexact[Array, " 3 N N (N//2+1) "],
) -> Inexact[Array, " 3 N N (N//2+1) "]:
    c1 = a[1] * b[2] - a[2] * b[1]
    c2 = a[2] * b[0] - a[0] * b[2]
    c3 = a[0] * b[1] - a[1] * b[0]
    return jnp.stack([c1, c2, c3], axis=0)


class ProjectedConvection3d(BaseNonlinearFun):
    derivative_operator: Complex[Array, " 3 N N (N//2+1) "]
    leray_projection: Leray

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, " 3 N N (N//2+1) "],
        dealiasing_fraction: float = 2 / 3,
    ):
        """
        Performs a pseudo-spectral evaluation of the nonlinear convection term
        for the 3d incompressible Navier-Stokes equations in velocity
        formulation using the rotational form. In state space, it reads

        ```
            𝒩(u) = 𝒫(u × ω)
        ```

        with `u` the velocity, `ω = ∇ × u` the vorticity, and `𝒫` the Leray
        projection onto divergence-free fields.

        The incompressible Navier-Stokes equations read

        ```
            uₜ = -(u ⋅ ∇)u - ∇p + ν Δu
        ```

        The [Lamb vector identity](https://en.wikipedia.org/wiki/Lamb_vector)
        allows rewriting the convection term as

        ```
            (u ⋅ ∇)u = ∇(|u|²/2) + ω × u
        ```

        Substituting and rearranging gives

        ```
            uₜ = u × ω - ∇(|u|²/2 + p) + ν Δu
        ```

        where `u × ω = -(ω × u)`. The Leray projection `𝒫` projects onto the
        space of divergence-free fields by removing any gradient component
        (i.e., `𝒫(∇φ) = 0` for any scalar `φ`). Since both the pressure
        gradient `∇p` and the kinetic energy gradient `∇(|u|²/2)` are
        gradients of scalar fields, the projection annihilates them:

        ```
            𝒫(uₜ) = 𝒫(u × ω) + ν Δu
        ```

        Hence, the nonlinear term reduces to `𝒩(u) = 𝒫(u × ω)`, and the
        pressure never needs to be computed explicitly.

        The curl is computed in Fourier space, the cross product `u × ω` in
        physical space (pseudo-spectral), and the result is projected back
        in Fourier space.

        Based on https://arxiv.org/pdf/1602.03638

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`. Must be
            `3`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same.
        - `derivative_operator`: A complex array of shape `(3, ..., N//2+1)`
            that represents the derivative operator in Fourier space.
        - `dealiasing_fraction`: The fraction of the highest resolved modes
            that are not aliased. Defaults to `2/3` which corresponds to
            Orszag's 2/3 rule.
        """
        if num_spatial_dims != 3:
            raise ValueError(
                "ProjectedConvection3d only supports 3 spatial dimensions."
            )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            dealiasing_fraction=dealiasing_fraction,
        )

        self.derivative_operator = derivative_operator

        self.leray_projection = Leray(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            derivative_operator=derivative_operator,
        )

    def __call__(
        self,
        u_hat: Complex[Array, " 3 N N (N//2+1) "],
    ) -> Complex[Array, " 3 N N (N//2+1) "]:
        velocity_hat = u_hat
        curl_hat = _cross_product_3d(
            self.derivative_operator,
            velocity_hat,
        )

        curl = self.ifft(self.dealias(curl_hat))
        velocity = self.ifft(self.dealias(velocity_hat))

        convection = _cross_product_3d(
            velocity,
            curl,
        )

        convection_hat = self.fft(convection)

        convection_projected_hat = self.leray_projection(convection_hat)

        return convection_projected_hat


class ProjectedConvection3dKolmogorov(ProjectedConvection3d):
    injection: Complex[Array, "3 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        derivative_operator: Complex[Array, " 3 ... (N//2)+1"],
        dealiasing_fraction: float,
    ):
        """
        Performs a pseudo-spectral evaluation of the nonlinear convection term
        together with a Kolmogorov-like injection term for the 3d
        incompressible Navier-Stokes equations in velocity formulation. In
        state space, it reads

        ```
            𝒩(u) = 𝒫(u × ω) + f
        ```

        For details on the projected convection term, see
        `exponax.nonlin_fun.ProjectedConvection3d`. The forcing term has the
        form

        ```
            f₀ = γ sin(k (2π/L) x₁)

            f₁ = 0

            f₂ = 0
        ```

        i.e., energy of intensity `γ` is injected at wavenumber `k` in the
        first velocity channel varying over the second spatial axis. This
        forcing is naturally divergence-free because the forced component does
        not vary along its own direction.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`. Must be
            `3`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same.
        - `injection_mode`: The wavenumber `k` at which energy is injected.
        - `injection_scale`: The intensity `γ` of the injection term.
        - `derivative_operator`: A complex array of shape `(3, ..., N//2+1)`
            that represents the derivative operator in Fourier space.
        - `dealiasing_fraction`: The fraction of the highest resolved modes
            that are not aliased. Defaults to `2/3` which corresponds to
            Orszag's 2/3 rule.
        """
        super().__init__(
            num_spatial_dims,
            num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=dealiasing_fraction,
        )

        wavenumbers = build_wavenumbers(num_spatial_dims, num_points)
        injection_mask = (
            (wavenumbers[0] == 0)
            & (wavenumbers[1] == injection_mode)
            & (wavenumbers[2] == 0)
        )
        # In 3D, we work with velocity directly (not vorticity), so the
        # forcing is f = γ sin(k x₁) ê₀. Only the first velocity channel is
        # forced, and no extra -k factor is needed (unlike the 2D vorticity
        # formulation).
        injection_single = jnp.where(
            injection_mask,
            injection_scale
            * build_scaling_array(num_spatial_dims, num_points, mode="coef_extraction"),
            0.0,
        )
        # Shape (3, N, N, N//2+1): forcing only in the first velocity channel
        zeros = jnp.zeros_like(injection_single)
        self.injection = jnp.concatenate([injection_single, zeros, zeros], axis=0)

    def __call__(
        self, u_hat: Complex[Array, "3 ... (N//2)+1"]
    ) -> Complex[Array, "3 ... (N//2)+1"]:
        neg_convection_hat = super().__call__(u_hat)
        return neg_convection_hat + self.injection
