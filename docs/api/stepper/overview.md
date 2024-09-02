# Overview

`Exponax` comes with many pre-built dynamics. They are all based on the idea of
solving semi-linear PDEs

$$
\partial_t u = \mathcal{L} u + \mathcal{N}(u)
$$

where $\mathcal{L}$ is a linear operator and $\mathcal{N}$ is a non-linear
differential operator.

TODO: add classification of dynamics based on:

- linear or nonlinear or reaction-diffusion
- specific (i.e. concrete equations like advection, Burgers etc.) or generic
  based on providing a collection of linear and nonlinear coefficients
- if generic: whether it uses a physical, normalized or difficulty-based
  interface
- in higher dimensions: whether it behaves isotropic or anisotropic (with
  different coefficients per direction)

Then the resulting behavior can also be classified into ...