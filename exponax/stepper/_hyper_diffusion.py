from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import ZeroNonlinearFun


class HyperDiffusion(BaseStepper):
    hyper_diffusivity: float
    diffuse_on_diffuse: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        hyper_diffusivity: float = 0.0001,
        diffuse_on_diffuse: bool = False,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) hyper-diffusion
        equation on periodic boundary conditions. A hyper-diffusion equation
        acts like a diffusion equation but higher wavenumbers/modes are damped
        even faster.

        In 1d, the hyper-diffusion equation is given by

        ```
            uₜ = - μ uₓₓₓₓ
        ```

        with `μ ∈ ℝ` being the hyper-diffusivity.

        Note the minus sign because by default, a fourth-order derivative
        dampens with a negative coefficient. To match the concept of
        second-order diffusion, a negation is introduced.

        In higher dimensions, the hyper-diffusion equation can be written as

        ```
            uₜ = − μ ((∇⊙∇) ⋅ (∇⊙∇)) u
        ```

        or

        ```
            uₜ = - μ Δ(Δu)
        ```

        The latter introduces spatial mixing.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `hyper_diffusivity` (keyword-only): The hyper-diffusivity `ν`.
            This stepper only supports scalar (=isotropic) hyper-diffusivity.
            Default: 0.0001.
        - `diffuse_on_diffuse` (keyword-only): If `True`, the second form
            of the hyper-diffusion equation in higher dimensions is used. As a
            consequence, there will be mixing in the spatial derivatives.
            Default: `False`.

        **Notes:**

        - The stepper is unconditionally stable, no matter the choice of
            any argument because the equation is solved analytically in Fourier
            space.
        - Ultimately, only the factor `μ Δt / L⁴` affects the characteristic
            of the dynamics. See also
            [`exponax.stepper.generic.NormalizedLinearStepper`][] with
            `normalized_coefficients = [0, 0, 0, 0, alpha_4]` with `alpha_4 = -
            hyper_diffusivity * dt / domain_extent**4`.
        """
        self.hyper_diffusivity = hyper_diffusivity
        self.diffuse_on_diffuse = diffuse_on_diffuse
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        # Use minus sign to have diffusion work in "correct direction" by default
        if self.diffuse_on_diffuse:
            laplace_operator = build_laplace_operator(derivative_operator)
            linear_operator = (
                -self.hyper_diffusivity * laplace_operator * laplace_operator
            )
        else:
            linear_operator = -self.hyper_diffusivity * build_laplace_operator(
                derivative_operator, order=4
            )

        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
        )
