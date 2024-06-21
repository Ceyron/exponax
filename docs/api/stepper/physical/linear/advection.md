# Advection

In 1D:

$$ \frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0 $$

In higher dimensions:

$$ \frac{\partial u}{\partial t} + \vec{c} \cdot \nabla u = 0 $$

(often just $\vec{c} = c \vec{1}$)


::: exponax.stepper.Advection
    options:
        members:
            - __init__
            - __call__