# ETDRK Backbone

Core clases that implement the Exponential Time Differencing Runge-Kutta (ETDRK)
method for solving semi-linear PDEs in form of timesteppers. Require supplying
the time step size $\Delta t$, the linear operator in Fourier space $\hat{\mathcal{L}}_h$, and the non-linear operator in Fourier space $\hat{\mathcal{N}}_h$.

::: exponax.etdrk.ETDRK0
    options:
        members:
            - __init__
            - step_fourier

---

::: exponax.etdrk.ETDRK1
    options:
        members:
            - __init__
            - step_fourier

---

::: exponax.etdrk.ETDRK2
    options:
        members:
            - __init__
            - step_fourier

---

::: exponax.etdrk.ETDRK3
    options:
        members:
            - __init__
            - step_fourier

---

::: exponax.etdrk.ETDRK4
    options:
        members:
            - __init__
            - step_fourier

---

::: exponax.etdrk.BaseETDRK
    options:
        members:
            - __init__
            - step_fourier

---

::: exponax.etdrk.roots_of_unity