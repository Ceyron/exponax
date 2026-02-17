# Design Decisions

The `Exponax` package targets the fixed time step simulation of semi-linear
partial differential equations. Those are PDEs of the form

$$
\partial_t u = \mathcal{L} u + \mathcal{N}(u)
$$

where $\mathcal{L}$ is a linear differential operator and $\mathcal{N}$ is a
non-linear differential operator. The equations are first-order in time.
Semi-linear means that the order of derivative in the linear operator
$\mathcal{L}$ is higher than the order of the non-linear operator $\mathcal{N}$.
Or in other terms, the difficulty in (numerically) solving this PDE mainly stems
from the linear part. Crucially, the linear operator $\mathcal{L}$ fully
diagonalizes in Fourier space, which is the key property exploited by the ETDRK
methods.

If a problem aligns with these constraints, Fourier pseudo-spectral ETDRK is
among the most efficient approaches for semi-linear PDEs on periodic domains.
The tensor-based operations integrate well with GPU execution and JAX's
automatic differentiation.

The package makes the following design decisions. Reasons can be (M)athematical
or (C)onvenience.

## Inherent Limitations of the Method

These constraints arise from the Fourier pseudo-spectral ETDRK approach itself.

## Periodic Boundary Conditions (M, C)

* Allows for usage of Fourier (pseudo-)spectral methods.
* The linear operator fully diagonalizes in Fourier space.
* FFTs are highly efficient (on the GPU).

## First-Order in Time (M)

* ETDRK methods are designed for PDEs that are first-order in time. Higher-order
  time derivatives (e.g., wave-type equations with $\partial_{tt} u$) do not
  directly conform to the framework.
* Reformulating higher-order PDEs into first-order systems often introduces
  channel mixing in the linear operator, which breaks the diagonal structure in
  Fourier space and makes the method inapplicable. (Though one can use clever
  diagonalization tricks but this is problem-specific; see
  [`Wave`](../api/stepper/linear/wave.md) for an example.)

## No Channel Mixing in the Linear Operator (M, C)

* Breaks down the diagonalization in Fourier space. (Though one can use clever
  diagonalization tricks but this is problem-specific; see
  [`Wave`](../api/stepper/linear/wave.md) for an example.)

## No Inhomogeneous Coefficients in Front of the Linear Operator (M)

* Also breaks down the diagonalization in Fourier space.
* Implement a custom nonlinear operator if you need inhomogeneous coefficients (which might then be subject to $\Delta t$ restrictions and requires the coefficient array to be sufficiently smooth. See also [this issue](https://github.com/tum-pbs/apebench/issues/40) for a discussion.)

## Only Smooth Problems (M)

* The method assumes smooth and bandlimited solutions whose Fourier spectrum
  decays rapidly at high frequencies.
* Precludes simulation of strongly hyperbolic PDEs with discontinuities such as
  the inviscid Burgers equation, Euler equations, or shallow water equations.
* Can handle their viscous counterparts (e.g., viscous Burgers, Navier-Stokes)
  where viscosity dampens high-frequency modes.
* This is an inherent limitation of Fourier pseudo-spectral methods, not just
  `Exponax`.

## Difficulty from the Linear Part (M)

* ETDRK methods treat the linear part analytically but only approximate the
  nonlinear part via a Runge-Kutta scheme.
* When the nonlinear part dominates (e.g., Navier-Stokes at high Reynolds
  numbers), the advantage of the exact linear treatment diminishes and small time
  steps become necessary. In such cases, use a [RepeatedStepper](../api/utilities/repeated_stepper.md) with a small internal time step.

## Implementation Choices

These constraints are specific to Exponax's implementation.

## The Domain Is Always a (Scaled) Hypercube (C)

The domain is always limited to $\Omega = (0, L)^D$ where $D$ is the dimension,
i.e., the extent is the same in all directions. In other words, the package
cannot simulate phenomena with an aspect ratio different from 1.

## Uniform Cartesian Grid (C, M)

* The number of discretization points $N$ is uniform across all spatial
  dimensions.
* Uniform spacing enables the use of FFTs for computing spectral derivatives.
* Simplifies derivative operator construction in Fourier space.

## Only Real-Valued PDEs (C)

Both the linear and the non-linear operator are real-valued.

* We can use `rfftn` by default which produces arrays of shape `(C, ...,
  N//2+1)`, saving about half the computation.
* Avoids ambiguities with spectral derivatives at the Nyquist mode.
* The evolved trajectory in state space is always real which more closely
  matches what deep learning typically expects.

## Fixed Time Step (C)

* Although ETDRK methods theoretically support adaptive time stepping, a
  constant $\Delta t$ simplifies the interface.
* All ETDRK coefficients can be precomputed once at initialization rather than
  recomputed at every step.
* Aligns with `jax.lax.scan` for efficient temporal rollouts.
* Fits many deep learning use cases where a fixed temporal step is embedded in
  the architecture.

## Most Pre-Defined Steppers Have Isotropic Linear Operators (C)

* Eases the interface. See the [stepper overview](../api/stepper/overview.md)
  for the full list of available steppers. Some steppers like
  [Advection](../api/stepper/linear/advection.md) and
  [Diffusion](../api/stepper/linear/diffusion.md) allow for anisotropy in higher
  dimensions.
* You can implement your own custom time stepper. `Exponax` supports anisotropy
  (=spatial mixing) but does **not** support channel mixing in the linear
  operator. However, channel mixing in the non-linear operator is fine!

## The Default Order of ETDRK Method Is 2 (C, M)

* The best compromise between computation cost, memory consumption, and
  numerical stability was observed when using single precision floats (the
  default in JAX). Higher-order integrators can be used to achieve high accurary when paired with double precision (activate via `jax.config.update("jax_enable_x64", True)`).
* See the [ETDRK backbone](../api/etdrk_backbone.md) for details on the
  available ETDRK orders.

## All Time-Steppers Are by Default Single-Batch (C)

In contrast to other deep learning frameworks (like PyTorch, TensorFlow, or
Flax), `Exponax` time steppers by default operate on tensors of the shape `(C,
*N)` with an arbitrary number of spatial axes `*N` and one leading channel axis.
Each timestepper also enforces the input to be of that shape. If you want to
operate on multiple states in batch use `jax.vmap` on them. This follows the
[Equinox](https://github.com/patrick-kidger/equinox) philosophy.

* Allows for tighter composition with other function transformations. For
  example, when doing a temporal rollout one can either do
  `rollout(jax.vmap(stepper), T)(u_0)` or `jax.vmap(rollout(stepper, T))(u_0)`.
  The former produces a trajectory of shape `(T, B, C, *N)` and the latter
  produces a trajectory of shape `(B, T, C, *N)` (i.e., the batch `B` and time
  `T` axes are swapped).

## There Are No Custom Grid or State Classes (C)

* Lean design that only focuses on JAX Arrays and PyTrees allows for tighter
  integration with other libraries in the JAX ecosystem.

## There Is No `jax.jit` Being (Explicitly) Used in the Package (C)

* `jit` is supposed to be user-facing functionality. Most of `Exponax`
  operations work fine under `jit`-transformation.
* However, if you use `exponax.repeat`, `exponax.rollout`, or
  `exponax.RepeatedStepper`, those internally use `jax.lax.scan` which is
  already a JIT-compiled loop.

## There Are Only Limited Shipped Visualization Routines (C)

* Keeps the package lean and focused on the core functionality.
* Visualization is very personal and problem-specific.
