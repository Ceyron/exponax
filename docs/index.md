# Getting Started

<p align="center">
  <img src="imgs/logo.svg" alt="exponax logo" width="50%">
</p>

`Exponax` is a suite for building Fourier spectral ETDRK time-steppers for
semi-linear PDEs in 1d, 2d, and 3d. There are many pre-built dynamics and plenty
of helpful utilities. It is extremely efficient, is differentiable (due to being
fully written in JAX), and embeds seamlessly into deep learning.

## Installation

```bash
pip install exponax
```

Requires Python 3.10+ and JAX 0.4.13+. 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quickstart

1d Kuramoto-Sivashinsky Equation.

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

For a next step, check out [this tutorial on 1D
Advection](https://fkoehler.site/exponax/examples/simple_advection_example_1d/)
that explains the basics of `Exponax`.

## Features


1. **JAX** as the computational backend:
    1. **Backend agnostic code** - run on CPU, GPU, or TPU, in both single and
        double precision.
    2. **Automatic differentiation** over the timesteppers - compute gradients
        of solutions with respect to initial conditions, parameters, etc.
    3. Also helpful for **tight integration with Deep Learning** since each
        timestepper is just an
        [Equinox](https://github.com/patrick-kidger/equinox) Module.
    4. **Automatic Vectorization** using `jax.vmap` (or `equinox.filter_vmap`)
        allowing to advance multiple states in time or instantiate multiple
        solvers at a time that operate efficiently in batch.
2. **Lightweight Design** without custom types. There is no `grid` or `state`
    object. Everything is based on `jax.numpy` arrays. Timesteppers are callable
    PyTrees.
3. More than 46 pre-built dynamics:
    1. Linear PDEs in 1d, 2d, and 3d (advection, diffusion, dispersion, etc.)
    2. Nonlinear PDEs in 1d, 2d, and 3d (Burgers, Kuramoto-Sivashinsky,
        Korteweg-de Vries, Navier-Stokes, etc.)
    3. Reaction-Diffusion (Gray-Scott, Swift-Hohenberg, etc.)
4. Collection of initial condition distributions (truncated Fourier series,
   Gaussian Random Fields, etc.)
5. **Utilities** for spectral derivatives, grid creation, autoregressive rollout,
   etc.
6. Easily extendable to new PDEs by subclassing from the `BaseStepper` module.
7. Normalized interface for reduced number of parameters to uniquely define any
   dynamics.

## Citation

This package was developed as part of the [APEBench paper
(arxiv.org/abs/2411.00180)](https://arxiv.org/abs/2411.00180) (accepted at
Neurips 2024). If you find it useful for your research, please consider citing
it:

```bibtex
@article{koehler2024apebench,
  title={{APEBench}: A Benchmark for Autoregressive Neural Emulators of {PDE}s},
  author={Felix Koehler and Simon Niedermayr and R{\"}udiger Westermann and Nils Thuerey},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={38},
  year={2024}
}
```

(Feel free to also give the project a star on GitHub if you like it.)

[Here](https://github.com/tum-pbs/apebench) you can find the APEBench benchmark suite.

## License

MIT, see [here](https://github.com/Ceyron/apebench/blob/main/LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler)
