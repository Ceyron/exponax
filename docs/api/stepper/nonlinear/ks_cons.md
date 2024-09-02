# Kuramoto-Sivashinsky (conservative format)

Uses the convection nonlinearity similar to Burgers, but only works in 1D:

$$ \frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial u^2}{\partial x} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} = 0 $$

::: exponax.stepper.KuramotoSivashinskyConservative
    options:
        members:
            - __init__
            - __call__