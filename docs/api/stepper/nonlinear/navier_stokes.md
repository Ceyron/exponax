# Navier-Stokes

## 2D: Streamfunction-Vorticity Formulation

In 2D, the vorticity is a scalar, so the streamfunction-vorticity formulation
reduces the system to a single-channel PDE — making it the natural choice for
2D spectral methods.

::: exponax.stepper.NavierStokesVorticity
    options:
        members:
            - __init__
            - __call__

---

::: exponax.stepper.KolmogorovFlowVorticity
    options:
        members:
            - __init__
            - __call__

---

## 3D: Velocity Formulation with Leray Projection

In 3D, the vorticity is also a three-component vector, offering no reduction in
degrees of freedom. The velocity formulation with Leray projection is preferred
instead, using the rotational form of the convection term.

::: exponax.stepper.NavierStokesVelocity
    options:
        members:
            - __init__
            - __call__

---

::: exponax.stepper.KolmogorovFlowVelocity
    options:
        members:
            - __init__
            - __call__
