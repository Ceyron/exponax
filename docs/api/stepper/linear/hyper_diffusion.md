# Hyper-Diffusion

In 1D:

$$ \frac{\partial u}{\partial t} = \xi \frac{\partial^4 u}{\partial x^4} $$

In higher dimensions:

$$ \frac{\partial u}{\partial t} = \zeta \nabla \cdot (\nabla \odot \nabla \odot \nabla) u $$

or with spatial mixing:

$$ \frac{\partial u}{\partial t} = \zeta (\nabla \cdot \nabla)(\nabla \cdot \nabla) u $$

::: exponax.stepper.HyperDiffusion
    options:
        members:
            - __init__
            - __call__