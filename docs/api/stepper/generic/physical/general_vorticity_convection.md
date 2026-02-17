# General Vorticity Convection Stepper

!!! note "2D only"

    This generic stepper uses the streamfunction-vorticity formulation and only
    works in 2D. For 3D incompressible Navier-Stokes, use
    [`NavierStokesVelocity`](../../nonlinear/navier_stokes.md) or
    [`KolmogorovFlowVelocity`](../../nonlinear/navier_stokes.md) which use the
    velocity formulation with Leray projection.

::: exponax.stepper.generic.GeneralVorticityConvectionStepper
    options:
        members:
            - __init__
            - __call__