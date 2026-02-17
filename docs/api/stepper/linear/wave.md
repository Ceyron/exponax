# Wave

In 1D:

$$ \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2} $$

In higher dimensions:

$$ \frac{\partial^2 u}{\partial t^2} = c^2 \Delta u $$

with $c \in \R$ the speed of sound (wave speed).

Internally, this second-order equation is rewritten as a first-order system of
two coupled fields — height $h$ and velocity $v = h_t$:

$$
h_t = v, \quad v_t = c^2 \Delta h
$$

The state therefore has **two channels**: `u[0]` is the height field $h$ and
`u[1]` is the velocity field $v$.

::: exponax.stepper.Wave
    options:
        members:
            - __init__
            - __call__
            - _forward_transform
            - _inverse_transform
            - step_fourier
