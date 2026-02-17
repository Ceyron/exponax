import jax.numpy as jnp
from jaxtyping import Array, Complex

from .._spectral import build_laplace_operator
from ._base import BaseNonlinearFun


class Leray(BaseNonlinearFun):
    inv_laplacian: Complex[Array, " 1 ... (N//2+1) "]
    derivative_operator: Complex[Array, " D ... (N//2+1) "]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        derivative_operator: Complex[Array, " D ... (N//2+1) "],
        order: int = 2,
    ):
        """
        Leray projection operator that projects a vector field onto the space of
        divergence-free fields. In state space, it reads

        ```
            𝒫(u) = u - ∇(Δ⁻¹ ∇ ⋅ u)
        ```

        This is equivalent to removing the irrotational (curl-free) component
        via the Helmholtz decomposition. The projection is trivial to compute in
        Fourier space because the Laplacian is diagonal.

        **Derivation:**

        Consider an operator splitting approach to the incompressible
        Navier-Stokes equations in which advection and diffusion have already
        been integrated, yielding an intermediate velocity `u*` that is
        generally **not** divergence-free (`∇ ⋅ u* ≠ 0`). The pressure substep
        reads

        ```
            (u** - u*) / Δt = -(1/ρ) ∇p
        ```

        We require `u**` to be incompressible (`∇ ⋅ u** = 0`). Applying the
        divergence to both sides gives

        ```
            (1/Δt)(∇ ⋅ u** - ∇ ⋅ u*) = -(1/ρ) Δp
        ```

        Setting `∇ ⋅ u** = 0` yields the pressure-Poisson equation

        ```
            Δp = (ρ/Δt) ∇ ⋅ u*
        ```

        Solving for `p` and substituting back gives

        ```
            u** = u* - (Δt/ρ) ∇(Δ⁻¹ (ρ/Δt) ∇ ⋅ u*)
        ```

        For constant density (as in the incompressible case), `ρ` and `Δt`
        cancel, and we obtain the Leray projection

        ```
            u** = u* - ∇(Δ⁻¹(∇ ⋅ u*))
        ```

        Hence, making a velocity field incompressible is a three-step process:

        1. Compute the divergence: `d = ∇ ⋅ u*`
        2. Solve a Poisson equation for a pseudo-pressure: `p = Δ⁻¹(-d)`
        3. Correct the velocity via the pressure gradient: `u** = u* + ∇p`

        In general, step 2 is the most computationally demanding (e.g.,
        requiring a conjugate gradient solve). However, in Fourier space the
        Laplacian is diagonal, making the Poisson solve a trivial pointwise
        division.

        Note that one can either use a negative sign in the Poisson solve (`p =
        Δ⁻¹(-d)`) or a positive sign (`p = Δ⁻¹(d)`) as long as the opposite sign
        is used in the velocity correction step. The choice of sign is arbitrary
        and does not affect the final result.

        Technically not a nonlinear operator, but it performs channel mixing
        between the velocity channels and hence subclasses `BaseNonlinearFun`.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `D`.
        - `num_points`: The number of points `N` used to discretize the domain.
            This **includes** the left boundary point and **excludes** the right
            boundary point. In higher dimensions; the number of points in each
            dimension is the same.
        - `derivative_operator`: A complex array of shape `(D, ..., N//2+1)`
            that represents the derivative operator in Fourier space.
        - `order`: The order of the Laplacian used for the Poisson solve.
            Default is `2`.
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
        )

        laplace_operator = build_laplace_operator(derivative_operator, order=order)

        self.inv_laplacian = jnp.where(
            laplace_operator != 0, 1.0 / laplace_operator, 0.0
        )

        self.derivative_operator = derivative_operator

    def __call__(
        self,
        u_hat: Complex[Array, " D ... (N//2+1) "],
    ) -> Complex[Array, " D ... (N//2+1) "]:
        """Compute the Leray projection of the velocity field u_hat.

        Args:
            u_hat: Velocity field in Fourier space.

        Returns:
            The Leray projection of u_hat in Fourier space.
        """
        div_u_hat = jnp.sum(
            self.derivative_operator * u_hat,
            axis=0,
            keepdims=True,
        )

        pressure_hat = -self.inv_laplacian * div_u_hat

        grad_pressure_hat = self.derivative_operator * pressure_hat

        return u_hat + grad_pressure_hat
