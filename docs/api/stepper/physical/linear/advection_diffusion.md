# Advection-Diffusion

In 1D:

$$ \frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$

In higher dimensions:

$$ \frac{\partial u}{\partial t} + \vec{c} \cdot \nabla u = \nu \nabla \cdot \nabla u $$

(often just $\vec{c} = c \vec{1}$) and potentially with anisotropic diffusion.

::: exponax.stepper.AdvectionDiffusion
    options:
        members:
            - __init__
            - __call__