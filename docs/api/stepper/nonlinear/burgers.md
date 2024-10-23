# Burgers

In 1D:

$$ \frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial u^2}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$

In higher dimensions:

$$ \frac{\partial u}{\partial t} + \frac{1}{2} \nabla \cdot (u \odot u) = \nu \nabla \cdot \nabla u $$

(with as many channels (=velocity components) as spatial dimensions)

::: exponax.stepper.Burgers
    options:
        members:
            - __init__
            - __call__