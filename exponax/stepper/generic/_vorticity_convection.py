import jax.numpy as jnp
from jaxtyping import Array, Complex

from ..._base_stepper import BaseStepper
from ...nonlin_fun import VorticityConvection2d, VorticityConvection2dKolmogorov


class GeneralVorticityConvectionStepper(BaseStepper):
    vorticity_convection_scale: float
    linear_coefficients: tuple[float, ...]
    injection_mode: int
    injection_scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        vorticity_convection_scale: float = 1.0,
        linear_coefficients: tuple[float, ...] = (0.0, 0.0, 0.001),
        injection_mode: int = 4,
        injection_scale: float = 0.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for 2D PDEs consisting of vorticity convection term and an
        arbitrary combination of (isotropic) linear derivatives.

        ```
            uₜ + b ([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u = sum_j a_j (1⋅∇ʲ)u
        ```

        where `b` is the vorticity convection scale, `a_j` are the coefficients
        of the linear derivatives, and `∇ʲ` is the `j`-th derivative operator.

        In the default configuration, this corresponds to the 2D Navier-Stokes
        simulation with a viscosity of `ν = 0.001` (the resulting Reynols number
        depends on the `domain_extent`. In the case of a unit square domain,
        i.e., `domain_extent = 1`, the Reynols number is `Re = 1/ν = 1000`).
        Depending on the initial state, this corresponds to a decaying 2D
        turbulence.

        Additionally, one can set an `injection_mode` and `injection_scale` to
        inject energy into the system. For example, this allows for the
        simulation of forced turbulence (=Kolmogorov flow).

        **Arguments:**

        - `num_spatial_dims`: number of spatial dimensions `D`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `linear_coefficients`: The list of coefficients `a_j`
            corresponding to the derivatives. The length of this tuple
            represents the highest occuring derivative. The default value `(0.0,
            0.0, 0.001)` corresponds to pure regular diffusion.
        - `vorticity_convection_scale`: The scale `b` of the vorticity
            convection term.
        - `injection_mode`: The mode of the injection.
        - `injection_scale`: The scale of the injection. Defaults to `0.0` which
            means no injection. Hence, the flow will decay over time.
        - `dealiasing_fraction`: The fraction of the modes that are kept after
            dealiasing. The default value `2/3` corresponds to the 2/3 rule.
        - `order`: The order of the ETDRK method to use. Must be one of {0, 1,
            2, 3, 4}. The option `0` only solves the linear part of the
            equation. Hence, only use this for linear PDEs. For nonlinear PDEs,
            a higher order method tends to be more stable and accurate. `2` is
            often a good compromis in single-precision. Use `4` together with
            double precision (`jax.config.update("jax_enable_x64", True)`) for
            highest accuracy.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method.
        """
        if num_spatial_dims != 2:
            raise ValueError(f"Expected num_spatial_dims = 2, got {num_spatial_dims}.")
        self.vorticity_convection_scale = vorticity_convection_scale
        self.linear_coefficients = linear_coefficients
        self.injection_mode = injection_mode
        self.injection_scale = injection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        linear_operator = sum(
            jnp.sum(
                c * (derivative_operator) ** i,
                axis=0,
                keepdims=True,
            )
            for i, c in enumerate(self.linear_coefficients)
        )
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> VorticityConvection2d:
        if self.injection_scale == 0.0:
            return VorticityConvection2d(
                self.num_spatial_dims,
                self.num_points,
                convection_scale=self.vorticity_convection_scale,
                derivative_operator=derivative_operator,
                dealiasing_fraction=self.dealiasing_fraction,
            )
        else:
            return VorticityConvection2dKolmogorov(
                self.num_spatial_dims,
                self.num_points,
                convection_scale=self.vorticity_convection_scale,
                derivative_operator=derivative_operator,
                dealiasing_fraction=self.dealiasing_fraction,
                injection_mode=self.injection_mode,
                injection_scale=self.injection_scale,
            )
