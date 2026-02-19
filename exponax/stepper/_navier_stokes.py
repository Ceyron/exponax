from jaxtyping import Array, Complex

from .._base_stepper import BaseStepper
from .._spectral import build_laplace_operator
from ..nonlin_fun import (
    ProjectedConvection3d,
    ProjectedConvection3dKolmogorov,
    VorticityConvection2d,
    VorticityConvection2dKolmogorov,
)


class NavierStokesVorticity(BaseStepper):
    diffusivity: float
    vorticity_convection_scale: float
    drag: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
        vorticity_convection_scale: float = 1.0,
        drag: float = 0.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the 2d Navier-Stokes equation on periodic boundary
        conditions in streamfunction-vorticity formulation. The equation reads

        ```
            uₜ + b ([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u = λu + ν Δu
        ```

        with `u` the vorticity. On the right-hand side the first term is a drag
        with coefficient `λ` and the second term is a diffusion with coefficient
        `ν`. The operation on the left-hand-side `([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u` is
        the "vorticity" convection which is scaled by `b`. It consists of the
        solution to the Poisson problem via the inverse Laplacian `Δ⁻¹` and the
        gradient `∇` of the streamfunction. The term `[1, -1]ᵀ ⊙` negates the
        second component of the gradient.

        We can map the vorticity to a (two-channel) velocity field by `∇
        (Δ⁻¹u)`.

        The expected temporal behavior is that the initial vorticity field
        continues to swirl but decays over time.

        Let `U = ‖∇ (Δ⁻¹u)‖` denote the magnitude of the velocity field, then
        the Reynolds number of the problem is `Re = U L / ν` with `L` the
        `domain_extent`.

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
        - `diffusivity`: The diffusivity coefficient `ν`. This affects the
            Reynolds number. The lower the diffusivity, the "more turbulent".
            Default is `0.01`.
        - `vorticity_convection_scale`: The scaling factor for the vorticity
            convection term. Default is `1.0`.
        - `drag`: The drag coefficient `λ`. Default is `0.0`.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for higher
            accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. The default 2/3 corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2. Default:
            2/3.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.

        **Notes:**

        - The Reynolds number is a measure of whether the problem is dominated
            by diffusive or convective effects. The higher the Reynolds number,
            the stronger the effect of the convective. Since this term is the
            nonlinear one, the higher the Reynolds number, the worse the ETDRK
            methods become in comparison to other approaches. That is because
            those methods are better for semi-linear PDEs in which the difficult
            part is the linear one.
        - The higher the Reynolds number, the smaller the timestep size must
            be to ensure stability.

        **Good Values:**

        - `domain_extent = 1`, `num_points=50`, `dt=0.01`,
            `diffusivity=0.0003`, together with an initial condition in which
            only the first few wavenumbers are excited gives a nice decaying
            turbulence demo.
        - Use the repeated stepper to perform 10 substeps to have faster
            dynamics.
        """
        if num_spatial_dims != 2:
            raise ValueError(
                f"Expected num_spatial_dims = 2, got {num_spatial_dims}. "
                "For 3D, use NavierStokesVelocity instead."
            )

        self.diffusivity = diffusivity
        self.vorticity_convection_scale = vorticity_convection_scale
        self.drag = drag
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
        return self.diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) + self.drag * build_laplace_operator(derivative_operator, order=0)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> VorticityConvection2d:
        return VorticityConvection2d(
            self.num_spatial_dims,
            self.num_points,
            convection_scale=self.vorticity_convection_scale,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
        )


class KolmogorovFlowVorticity(BaseStepper):
    diffusivity: float
    convection_scale: float
    drag: float
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
        diffusivity: float = 0.001,
        convection_scale: float = 1.0,
        drag: float = -0.1,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the 2d Kolmogorov flow equation on periodic boundary
        conditions in streamfunction-vorticity formulation. The equation reads

        ```
            uₜ + b ([1, -1]ᵀ ⊙ ∇(Δ⁻¹u)) ⋅ ∇u = λu + ν Δu + f
        ```

        For a detailed description of the terms, see the documentation of the
        `NavierStokesVorticity` stepper. The only difference is the additional
        forcing term `f` which injects new energy into the system. For a
        Kolmogorov flow **in primary variables**, it has the form

        ```
            f₀ = γ sin(k (2π/L) x₁)

            f₁ = 0
        ```

        In words, only the first channel is forced at a specific wavenumber over
        the second axis. Since this stepper considers the
        streamfunction-vorticity formulation, we take its curl to get

        ```
            f = -k (2π/L) γ cos(k (2π/L) x₁)
        ```

        The expected temporal behavior is that the initial vorticity field first
        is excited into a noisy striped pattern. This pattern breaks up and a
        turbulent spatio-temporal chaos emerges.

        A negative drag coefficient `λ` is needed to remove some of the energy
        piling up in low modes.

        According to

            Chandler, G.J. and Kerswell, R.R. (2013) ‘Invariant recurrent
            solutions embedded in a turbulent two-dimensional Kolmogorov flow’,
            Journal of Fluid Mechanics, 722, pp. 554–595.
            doi:10.1017/jfm.2013.122.

        equation (2.5), the Reynolds number of the Kolmogorov flow is given by

            Re = √ζ / ν √(L / (2π))³

        with `ζ` being the scaling of the Kolmogorov forcing, i.e., the
        `injection_scale`. Hence, in the case of `L = 2π`, `ζ = 1`, the Reynolds
        number is `Re = 1 / ν`. If one uses the default value of `ν = 0.001`,
        the Reynolds number is `Re = 1000` which also corresponds to the main
        experiments in

            Kochkov, D., Smith, J.A., Alieva, A., Wang, Q., Brenner, M.P. and
            Hoyer, S., 2021. Machine learning–accelerated computational fluid
            dynamics. Proceedings of the National Academy of Sciences, 118(21),
            p.e2101784118.

        together with `injection_mode = 4`. Note that they required a resolution
        of `num_points = 2048` (=> 2048^2 = 4.2M degrees of freedom in 2d) to
        fully resolve all scales at that Reynolds number. Using `Re = 0.01`
        which corresponds to `ν = 0.01` can be a good starting for
        `num_points=128`.


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
        - `diffusivity`: The diffusivity coefficient `ν`. This affects the
            Reynolds number. The lower the diffusivity, the "more turbulent".
            Default is `0.001`.
        - `convection_scale`: The scaling factor for the vorticity
            convection term. Default is `1.0`.
        - `drag`: The drag coefficient `λ`. Default is `-0.1`.
        - `injection_mode`: The mode of the injection. Default is `4`.
        - `injection_scale`: The scaling factor for the injection. Default
            is `1.0`.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for higher
            accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. The default 2/3 corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2. Default:
            2/3.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.

        **Notes:**

        - Interesting pointer to explore:
          http://trieste-conf.itp.ac.ru/Boffetta.pdf
        """
        if num_spatial_dims != 2:
            raise ValueError(
                f"Expected num_spatial_dims = 2, got {num_spatial_dims}. "
                "For 3D, use KolmogorovFlowVelocity instead."
            )
        self.diffusivity = diffusivity
        self.convection_scale = convection_scale
        self.drag = drag
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
        return self.diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) + self.drag * build_laplace_operator(derivative_operator, order=0)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> VorticityConvection2dKolmogorov:
        return VorticityConvection2dKolmogorov(
            self.num_spatial_dims,
            self.num_points,
            convection_scale=self.convection_scale,
            injection_mode=self.injection_mode,
            injection_scale=self.injection_scale,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
        )


class NavierStokesVelocity(BaseStepper):
    diffusivity: float
    drag: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
        drag: float = 0.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the 3d incompressible Navier-Stokes equations on
        periodic boundary conditions in velocity formulation. The equation
        reads

        ```
            uₜ = ν Δu + λu + 𝒫(u × ω)
        ```

        with `u` the three-channel velocity field and `ω = ∇ × u` the
        vorticity. The term `𝒫` denotes the Leray projection which enforces
        the incompressibility constraint `∇ ⋅ u = 0` by removing gradient
        components (pressure and kinetic energy gradient). The first term on
        the right-hand side is a diffusion with coefficient `ν` and the second
        term is an optional drag with coefficient `λ`.

        The nonlinear term uses the rotational form which exploits the vector
        identity `(u ⋅ ∇)u = ∇(|u|²/2) + ω × u` and the fact that the Leray
        projection annihilates all gradient fields.

        The Reynolds number of the problem is `Re = U L / ν` with `U` a
        characteristic velocity scale and `L` the `domain_extent`.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`. Must be
            `3`.
        - `domain_extent`: The size of the domain `L`; the domain is assumed
            to be a scaled hypercube `Ω = (0, L)³`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. The number of points in each dimension
            is the same. Hence, the total number of degrees of freedom is
            `N³`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `diffusivity`: The diffusivity (viscosity) coefficient `ν`. This
            affects the Reynolds number. The lower the diffusivity, the "more
            turbulent". Default is `0.01`.
        - `drag`: The drag coefficient `λ`. Default is `0.0`.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for
            higher accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. The default 2/3 corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2. Default:
            2/3.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.

        **Notes:**

        - In 3d, the velocity formulation is preferred over the vorticity
            formulation because the vorticity is also a three-component vector,
            offering no reduction in degrees of freedom. For 2d, use
            `NavierStokesVorticity` which solves for the scalar vorticity
            instead.
        - The nonlinear term uses the rotational form `𝒫(u × ω)` rather
            than the convective form `𝒫(-(u ⋅ ∇)u)`. Both are equivalent at
            the continuous level, but the rotational form is preferred for
            pseudo-spectral methods because it requires fewer FFTs. In the
            convective form, computing `(u ⋅ ∇)u` requires for each of the 3
            velocity components a dot product of `u` with its gradient, i.e.,
            `Σⱼ uⱼ ∂uᵢ/∂xⱼ`. That amounts to 3 × 3 = 9 physical-space
            multiplications, each requiring an inverse FFT for the operand and
            a forward FFT for the result. In the rotational form, the curl
            `ω = ∇ × u` is free in Fourier space (a cross product with `ik`),
            and the subsequent cross product `u × ω` in physical space
            involves only 6 multiplications (two per output component).
            Additionally, the gradient terms `∇(|u|²/2 + p)` are implicitly
            eliminated by the Leray projection without ever being computed.
            Since the FFTs are the most computationally demanding operations
            in higher dimensions (scaling as `O(N³ log N)` in 3d), reducing
            their count directly improves performance.
        - The higher the Reynolds number, the smaller the timestep size must
            be to ensure stability.
        """
        if num_spatial_dims != 3:
            raise ValueError(
                f"Expected num_spatial_dims = 3, got {num_spatial_dims}. "
                "For 2D, use NavierStokesVorticity instead."
            )

        self.diffusivity = diffusivity
        self.drag = drag
        self.dealiasing_fraction = dealiasing_fraction

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=3,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(self, derivative_operator):
        return self.diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) + self.drag * build_laplace_operator(derivative_operator, order=0)

    def _build_nonlinear_fun(self, derivative_operator):
        return ProjectedConvection3d(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
        )


class KolmogorovFlowVelocity(BaseStepper):
    diffusivity: float
    drag: float
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
        diffusivity: float = 0.01,
        drag: float = 0.0,
        injection_mode: int = 4,
        injection_scale: float = 1.0,
        order: int = 2,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Timestepper for the 3d Kolmogorov flow equation on periodic boundary
        conditions in velocity formulation. The equation reads

        ```
            uₜ = ν Δu + λu + 𝒫(u × ω) + f
        ```

        For a detailed description of the terms, see the documentation of the
        `NavierStokesVelocity` stepper. The only difference is the additional
        forcing term `f` which injects new energy into the system. It has the
        form

        ```
            f₀ = γ sin(k (2π/L) x₁)

            f₁ = 0

            f₂ = 0
        ```

        In words, only the first velocity channel is forced at a specific
        wavenumber over the second spatial axis. This forcing is divergence-
        free because the forced component does not vary along its own
        direction.

        The expected temporal behavior is that the velocity field develops
        shear layers which become unstable and break up into turbulent
        spatio-temporal chaos.

        A negative drag coefficient `λ` is needed to remove some of the energy
        piling up in low modes.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`. Must be
            `3`.
        - `domain_extent`: The size of the domain `L`; the domain is assumed
            to be a scaled hypercube `Ω = (0, L)³`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. The number of points in each dimension
            is the same. Hence, the total number of degrees of freedom is
            `N³`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `diffusivity`: The diffusivity (viscosity) coefficient `ν`. This
            affects the Reynolds number. The lower the diffusivity, the "more
            turbulent". Default is `0.01`.
        - `drag`: The drag coefficient `λ`. Default is `0.0`.
        - `injection_mode`: The wavenumber `k` at which energy is injected.
            Default is `4`.
        - `injection_scale`: The intensity `γ` of the injection term. Default
            is `1.0`.
        - `order`: The order of the Exponential Time Differencing Runge
            Kutta method. Must be one of {0, 1, 2, 3, 4}. The option `0` only
            solves the linear part of the equation. Use higher values for
            higher accuracy and stability. The default choice of `2` is a good
            compromise for single precision floats.
        - `dealiasing_fraction`: The fraction of the wavenumbers to keep
            before evaluating the nonlinearity. The default 2/3 corresponds to
            Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2. Default:
            2/3.
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.
        """
        if num_spatial_dims != 3:
            raise ValueError(
                f"Expected num_spatial_dims = 3, got {num_spatial_dims}. "
                "For 2D, use KolmogorovFlowVorticity instead."
            )
        self.diffusivity = diffusivity
        self.drag = drag
        self.injection_mode = injection_mode
        self.injection_scale = injection_scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=3,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        return self.diffusivity * build_laplace_operator(
            derivative_operator, order=2
        ) + self.drag * build_laplace_operator(derivative_operator, order=0)

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ProjectedConvection3dKolmogorov:
        return ProjectedConvection3dKolmogorov(
            self.num_spatial_dims,
            self.num_points,
            injection_mode=self.injection_mode,
            injection_scale=self.injection_scale,
            derivative_operator=derivative_operator,
            dealiasing_fraction=self.dealiasing_fraction,
        )
