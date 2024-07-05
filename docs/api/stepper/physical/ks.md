# Kuramoto-Sivashinsky equation

In 1D:

$$ \frac{\partial u}{\partial t} + \frac{1}{2} \left(\frac{\partial u}{\partial x}\right)^2 + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} = 0 $$

In higher dimensions:

$$ \frac{\partial u}{\partial t} + \frac{1}{2} \left \| \nabla u \right \|^2 + \nabla \cdot \nabla u + \nabla \cdot (\nabla \odot \nabla \odot \nabla) u = 0 $$

Uses the combustion format via the gradient norm that easily scales to higher dimensions.

::: exponax.stepper.KuramotoSivashinsky
    options:
        members:
            - __init__
            - __call__