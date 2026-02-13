<p align="center">
  <img src="docs/imgs/logo.svg" alt="exponax logo" width="300">
</p>
<h4 align="center">Efficient Differentiable n-d PDE solvers built on top of <a href="https://github.com/google/jax" target="_blank">JAX</a> & <a href="https://github.com/patrick-kidger/equinox" target="_blank">Equinox</a>.</h4>

<p align="center">
<a href="https://pypi.org/project/exponax/">
  <img src="https://img.shields.io/pypi/v/exponax.svg" alt="PyPI">
</a>
<a href="https://github.com/ceyron/exponax/actions/workflows/test.yml">
  <img src="https://github.com/ceyron/exponax/actions/workflows/test.yml/badge.svg" alt="Tests">
</a>
<a href="https://codecov.io/gh/Ceyron/exponax">
  <img src="https://codecov.io/gh/Ceyron/exponax/branch/main/graph/badge.svg" alt="codecov">
</a>
<a href="https://fkoehler.site/exponax/">
  <img src="https://img.shields.io/badge/docs-latest-green" alt="docs-latest">
</a>
<a href="https://github.com/ceyron/exponax/releases">
  <img src="https://img.shields.io/github/v/release/ceyron/exponax?include_prereleases&label=changelog" alt="Changelog">
</a>
<a href="https://github.com/ceyron/exponax/blob/main/LICENSE.txt">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#built-in-equations">Equations</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#background">Background</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/690a2faf-8d72-42b6-bae4-3ba6f4f75e88", width=700px>
</p>

`Exponax` solves partial differential equations in 1D, 2D, and 3D on periodic
domains highly efficiently using Fourier spectral methods and exponential time
differencing. It ships more than 46 PDE solvers covering linear, nonlinear, and
reaction-diffusion dynamics. Built entirely on
[JAX](https://github.com/google/jax) and
[Equinox](https://github.com/patrick-kidger/equinox), every solver is
automatically differentiable, JIT-compilable, and GPU/TPU-ready — making it
ideal for physics-based deep learning workflows.

## Installation

```bash
pip install exponax
```

Requires Python 3.10+ and JAX 0.4.13+. 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quickstart

Simulate the chaotic **Kuramoto-Sivashinsky equation** in 1D — a single stepper
object, one line to roll out 500 time steps:

```python
import jax
import exponax as ex
import matplotlib.pyplot as plt

ks_stepper = ex.stepper.KuramotoSivashinskyConservative(
    num_spatial_dims=1, domain_extent=100.0,
    num_points=200, dt=0.1,
)

u_0 = ex.ic.RandomTruncatedFourierSeries(
    num_spatial_dims=1, cutoff=5
)(num_points=200, key=jax.random.PRNGKey(0))

trajectory = ex.rollout(ks_stepper, 500, include_init=True)(u_0)

plt.imshow(trajectory[:, 0, :].T, aspect='auto', cmap='RdBu', vmin=-2, vmax=2, origin="lower")
plt.xlabel("Time"); plt.ylabel("Space"); plt.show()
```

![](https://github.com/user-attachments/assets/e4889898-9a74-4b6f-9e88-ee12706b2f6c)

Because every stepper is a differentiable JAX function, you can freely compose
it with `jax.grad`, `jax.vmap`, and `jax.jit`:

```python
# Jacobian of the stepper function
jacobian = jax.jacfwd(ks_stepper)(u_0)
```

For a next step, check out [this tutorial on 1D
Advection](https://fkoehler.site/exponax/examples/simple_advection_example_1d/)
that explains the basics of `Exponax`.

## Built-in Equations

### Linear

| Equation | Stepper | Dimensions |
|----------|---------|------------|
| Advection: $u_t + c \cdot \nabla u = 0$ | [`Advection`](https://fkoehler.site/exponax/api/stepper/linear/advection/) | 1D, 2D, 3D |
| Diffusion: $u_t = \nu \Delta u$ | [`Diffusion`](https://fkoehler.site/exponax/api/stepper/linear/diffusion/) | 1D, 2D, 3D |
| Advection-Diffusion: $u_t + c \cdot \nabla u = \nu \Delta u$ | [`AdvectionDiffusion`](https://fkoehler.site/exponax/api/stepper/linear/advection_diffusion/) | 1D, 2D, 3D |
| Dispersion: $u_t = \xi \nabla^3 u$ | [`Dispersion`](https://fkoehler.site/exponax/api/stepper/linear/dispersion/) | 1D, 2D, 3D |
| Hyper-Diffusion: $u_t = -\zeta \Delta^2 u$ | [`HyperDiffusion`](https://fkoehler.site/exponax/api/stepper/linear/hyper_diffusion/) | 1D, 2D, 3D |

### Nonlinear

| Equation | Stepper | Dimensions |
|----------|---------|------------|
| Burgers: $u_t + \frac{1}{2} \nabla \cdot (u \otimes u) = \nu \Delta u$ | [`Burgers`](https://fkoehler.site/exponax/api/stepper/nonlinear/burgers/) | 1D, 2D, 3D |
| Korteweg-de Vries: $u_t + \frac{1}{2} \nabla \cdot (u \otimes u) - \nabla^3 u = \mu \Delta u$ | [`KortewegDeVries`](https://fkoehler.site/exponax/api/stepper/nonlinear/kdv/) | 1D, 2D, 3D |
| Kuramoto-Sivashinsky: $u_t + \frac{1}{2} \|\nabla u\|^2 + \Delta u + \Delta^2 u = 0$ | [`KuramotoSivashinsky`](https://fkoehler.site/exponax/api/stepper/nonlinear/ks/) | 1D, 2D, 3D |
| KS (conservative): $u_t + \frac{1}{2} \nabla \cdot (u \otimes u) + \Delta u + \Delta^2 u = 0$ | [`KuramotoSivashinskyConservative`](https://fkoehler.site/exponax/api/stepper/nonlinear/ks_cons/) | 1D, 2D, 3D |
| Navier-Stokes (vorticity): $\omega_t + (u \cdot \nabla)\omega = \nu \Delta \omega$ | [`NavierStokesVorticity`](https://fkoehler.site/exponax/api/stepper/nonlinear/navier_stokes/) | 2D |
| Kolmogorov Flow (vorticity): $\omega_t + (u \cdot \nabla)\omega = \nu \Delta \omega + f$ | [`KolmogorovFlowVorticity`](https://fkoehler.site/exponax/api/stepper/nonlinear/navier_stokes/) | 2D |

### Reaction-Diffusion

| Equation | Stepper | Dimensions |
|----------|---------|------------|
| Fisher-KPP: $u_t = \nu \Delta u + r\, u(1 - u)$ | [`reaction.FisherKPP`](https://fkoehler.site/exponax/api/stepper/reaction/fisher_kpp/) | 1D, 2D, 3D |
| Allen-Cahn: $u_t = \nu \Delta u + c_1 u + c_3 u^3$ | [`reaction.AllenCahn`](https://fkoehler.site/exponax/api/stepper/reaction/allen_cahn/) | 1D, 2D, 3D |
| Cahn-Hilliard: $u_t = \nu \Delta(u^3 + c_1 u - \gamma \Delta u)$ | [`reaction.CahnHilliard`](https://fkoehler.site/exponax/api/stepper/reaction/cahn_hilliard/) | 1D, 2D, 3D |
| Gray-Scott: $u_t = \nu_1 \Delta u + f(1-u) - uv^2, \quad v_t = \nu_2 \Delta v - (f+k)v + uv^2$ | [`reaction.GrayScott`](https://fkoehler.site/exponax/api/stepper/reaction/gray_scott/) | 1D, 2D, 3D |
| Swift-Hohenberg: $u_t = ru - (k + \Delta)^2 u + g(u)$ | [`reaction.SwiftHohenberg`](https://fkoehler.site/exponax/api/stepper/reaction/swift_hohenberg/) | 1D, 2D, 3D |

<details>
<summary><strong>Generic stepper families</strong> (for advanced / custom dynamics)</summary>

These parametric families generalize the concrete steppers above. Each comes in
three flavors: physical coefficients, normalized, and difficulty-based.

| Family | Nonlinearity | Generalizes |
|--------|-------------|-------------|
| [`GeneralLinearStepper`](https://fkoehler.site/exponax/api/stepper/generic/physical/general_linear/) | None | Advection, Diffusion, Dispersion, etc. |
| [`GeneralConvectionStepper`](https://fkoehler.site/exponax/api/stepper/generic/physical/general_convection/) | Quadratic convection | Burgers, KdV, KS Conservative |
| [`GeneralGradientNormStepper`](https://fkoehler.site/exponax/api/stepper/generic/physical/general_gradient_norm/) | Gradient norm | Kuramoto-Sivashinsky |
| [`GeneralVorticityConvectionStepper`](https://fkoehler.site/exponax/api/stepper/generic/physical/general_vorticity_convection/) | Vorticity convection (2D only) | Navier-Stokes, Kolmogorov Flow |
| [`GeneralPolynomialStepper`](https://fkoehler.site/exponax/api/stepper/generic/physical/general_polynomial/) | Arbitrary polynomial | Fisher-KPP, Allen-Cahn, etc. |
| [`GeneralNonlinearStepper`](https://fkoehler.site/exponax/api/stepper/generic/physical/general_nonlinear/) | Convection + gradient norm + polynomial | Most of the above |

See the [normalized & difficulty interface docs](https://fkoehler.site/exponax/api/utilities/normalized_and_difficulty/) for details.

</details>

## Features

- **Hardware-agnostic** — run on CPU, GPU, or TPU in single or double precision.
- **Fully differentiable** — compute gradients of solutions w.r.t. initial conditions, PDE parameters, or neural network weights when composed with PDE solvers via `jax.grad`.
- **Vectorized batching** — advance multiple states or sweep over parameter grids in parallel using `jax.vmap` (and `eqx.filter_vmap`).
- **Deep-learning native** — every stepper is an [Equinox](https://github.com/patrick-kidger/equinox) Module, composable with neural networks out of the box.
- **Lightweight design** — no custom grid or state objects; everything is plain `jax.numpy` arrays and callable PyTrees.
- **Initial conditions** — library of random IC distributions (truncated Fourier series, Gaussian random fields, etc.).
- **Utilities** — spectral derivatives, grid creation, autoregressive rollout, interpolation, and more.
- **Extensible** — add new PDEs by subclassing `BaseStepper`.

## Documentation

Documentation is available at [fkoehler.site/exponax](https://fkoehler.site/exponax/). Key pages:

- [1D Advection Tutorial](https://fkoehler.site/exponax/examples/simple_advection_example_1d/) — learn the basics
- [Solver Showcase 1D](https://fkoehler.site/exponax/examples/solver_showcase_1d/) / [2D](https://fkoehler.site/exponax/examples/solver_showcase_2d/) / [3D](https://fkoehler.site/exponax/examples/solver_showcase_3d/) — visual gallery of all dynamics
- [Creating Your Own Solvers](https://fkoehler.site/exponax/examples/creating_your_own_solvers_1d/) — extend Exponax with custom PDEs
- [Training a Neural Operator](https://fkoehler.site/exponax/examples/learning_burgers_autoregressive_neural_operator/) — use `Exponax` for synthetic data generation and training of a neural emulator
- [Stepper Overview](https://fkoehler.site/exponax/api/stepper/overview/) — API reference for all steppers
- [Performance Hints](https://fkoehler.site/exponax/examples/performance_hints/) — tips for fast simulations

## Background

Exponax solves semi-linear PDEs of the form

$$ \partial u / \partial t = Lu + N(u), $$

where $L$ is a linear differential operator and $N$ is a nonlinear differential
operator. The linear part is solved exactly via a matrix exponential in Fourier
space, while the nonlinear part is integrated using exponential time
differencing Runge-Kutta (ETDRK) schemes of order 1 through 4. The complex
contour integral method of Kassam & Trefethen is used for numerical stability.

By restricting to periodic domains on scaled hypercubes with uniform Cartesian
grids, all transforms reduce to FFTs — yielding blazing-fast simulations. For
example, 50 trajectories of the 2D Kuramoto-Sivashinsky equation (200 time
steps, 128x128 grid) are generated in under a second on a modern GPU.

<details>
<summary>References</summary>

1. Cox, S.M. and Matthews, P.C. "Exponential time differencing for stiff systems." *Journal of Computational Physics* 176.2 (2002): 430-455. [doi:10.1006/jcph.2002.6995](https://doi.org/10.1006/jcph.2002.6995)
2. Kassam, A.K. and Trefethen, L.N. "Fourth-order time-stepping for stiff PDEs." *SIAM Journal on Scientific Computing* 26.4 (2005): 1214-1233. [doi:10.1137/S1064827502410633](https://doi.org/10.1137/S1064827502410633)
3. Montanelli, H. and Bootland, N. "Solving periodic semilinear stiff PDEs in 1D, 2D and 3D with exponential integrators." *Mathematics and Computers in Simulation* 178 (2020): 307-327. [doi:10.1016/j.matcom.2020.06.008](https://doi.org/10.1016/j.matcom.2020.06.008)

</details>

## Related & Motivation

This package is greatly inspired by the
[`spinX`](https://www.chebfun.org/docs/guide/guide19.html) module of the
[ChebFun](https://www.chebfun.org/) package in *MATLAB*. `spinX` served as a
reliable data generator for early works in physics-based deep learning, e.g.,
[DeepHiddenPhysics](https://github.com/maziarraissi/DeepHPMs/tree/7b579dbdcf5be4969ebefd32e65f709a8b20ec44/Matlab)
and [Fourier Neural
Operators](https://github.com/neuraloperator/neuraloperator/tree/af93f781d5e013f8ba5c52baa547f2ada304ffb0/data_generation).
However, due to the two-language barrier, dynamically calling *MATLAB* solvers
from Python-based deep learning workflows is hard to impossible. This also
excludes the option to differentiate through them — ruling out
differentiable-physics approaches like solver-in-the-loop correction or
diverted-chain training.

We view `Exponax` as a spiritual successor of `spinX`. JAX, as the
computational backend, elevates the power of this solver type with automatic
vectorization (`jax.vmap`), backend-agnostic execution (CPU/GPU/TPU), and tight
integration for deep learning via its versatile automatic differentiation
engine. With reproducible randomness in JAX, datasets can be re-created in
seconds — no need to ever write them to disk.

Beyond ChebFun, other popular pseudo-spectral implementations include
[Dedalus](https://dedalus-project.org/) in the Python world and
[FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl) in the
*Julia* ecosystem (the latter was especially helpful for verifying our
implementation of the contour integral method and dealiasing).

## Citation

`Exponax` was developed as part of the
[APEBench](https://github.com/tum-pbs/apebench) benchmark suite for
autoregressive neural emulators of PDEs. The accompanying paper was accepted at
**NeurIPS 2024**. If you find this package useful for your research, please
consider citing it:

```bibtex
@article{koehler2024apebench,
  title={{APEBench}: A Benchmark for Autoregressive Neural Emulators of {PDE}s},
  author={Felix Koehler and Simon Niedermayr and R{\"u}diger Westermann and Nils Thuerey},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={38},
  year={2024}
}
```

If you enjoy the project, feel free to give it a star on GitHub!

## Funding

The main author (Felix Koehler) is a PhD student in the group of [Prof. Thuerey at TUM](https://ge.in.tum.de/) and his research is funded by the [Munich Center for Machine Learning](https://mcml.ai/).

## License

MIT, see [here](https://github.com/Ceyron/exponax/blob/main/LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler) &nbsp;&middot;&nbsp;
> LinkedIn [Felix Köhler](https://www.linkedin.com/in/felix-koehler)
