# Diffusion

In 1D:

$$ \frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2} $$

In higher dimensions:

$$ \frac{\partial u}{\partial t} = \nu \nabla \cdot \nabla u $$

or with anisotropic diffusion:

$$ \frac{\partial u}{\partial t} = \nabla \cdot \left( A \nabla u \right) $$

with $A \in \R^{D \times D}$ symmetric positive definite.

::: exponax.stepper.Diffusion
    options:
        members:
            - __init__
            - __call__