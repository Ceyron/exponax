# Dispersion

In 1D:

$$ \frac{\partial u}{\partial t} = \xi \frac{\partial^3 u}{\partial x^3} $$

In higher dimensions:

$$ \frac{\partial u}{\partial t} = \xi \nabla \cdot (\nabla \odot \nabla) u $$

or with spatial mixing:

$$ \frac{\partial u}{\partial t} = \xi (1 \cdot \nabla) (\nabla \cdot \nabla) u $$

::: exponax.stepper.Dispersion
    options:
        members:
            - __init__
            - __call__