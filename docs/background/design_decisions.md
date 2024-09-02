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
Or in other terms, the difficulty in (numerically) solving this PDE mainly stems from
the linear part.

The package incurs the following design decisions. Reasons can be (M)athematical
or (C)onvenience.

### Periodic Boundary Conditions (M, C)

* Allows for usage of Fourier (pseudo-)spectral methods.
* The linear operator fully diagonalizes in Fourier space.
* FFTs are highly efficient (on the GPU).

### The domain is always a (scaled) hypercube (C)

The domain is always limited to $\Omega = (0, L)^D$ where $D$ is the dimension,
i.e., the extent is the same in all directions. In other words, the package
cannot simulate phenomena with an aspect ratio different from 1.

### The domain is discretized with a uniform Cartesian grid with same number of degrees of freedom in each direction (C, M)

### Only real-valued PDEs (C)

Both the linear and the non-linear operator are real-valued.

* We can use the `rfftn` by default (saves about half the computation)
* Avoids ambiguities with spectral derivatives at the Nyquist mode
* The evolved trajectory in state space is always real which more closely
  matches what deep learning typically expects.

### No channel mixing in the linear operator (M, C)

* breaks down the diagonalization in Fourier space

### No inhomogeneous coefficients in front of the linear operator (M)

* also breaks down the diagonalization in Fourier space
* implement a custom nonlinear operator if you need inhomogeneous coefficients

### Fixed time step (C)

...

### Only smooth problems (M)

* The package does not support problems with discontinuities or shocks.

### Most pre-defined steppers have isotropic linear operators (C)

* Eases the interface
* One can implement its own custom time stepper. `Exponax` supports anistropy
  (=spatial mixing) but does **not** support channel mixing in the linear
  operator. However, channel mixing in the non-linear operator is fine!

### The default order of ETDRK method is 2 (C, M)

* I observed the best numerical stability in the coefficient computation.
  Higher-order methods have more sensible coefficient computation relying more
  greatly on the complex contour integral method.

### Works only with problems for which the difficulty stems from the linear part (M)

* This is noticeable that the benefit for example for Navier-Stokes at higher
  Reynolds numbers becomes less and less.

### All time-steppers are by default single-batch (C)

In contrast to other deep learning frameworks (like PyTorch, TensorFlow, or
Flax), `Exponax` time steppers by default operate of tensors of the shape `(C,
*N)` with an arbitrary number of spatial dimensions `*N` and one leading channel
dimension. Each timestepper also enforces the input to be of that shape. If you
want to operate on multiple states in batch use `jax.vmap` on them. This follows
the [Equinox](https://github.com/patrick-kidger/equinox) philosophy.

* Allows for tighter composition with other function transformations. For
  example, when doing a temporal rollout one can either do
  `rollout(jax.vmap(stepper), T)(u_0)` or `jax.vmap(rollout(stepper, T))(u_0)`.
  The former produces a trajectory of shape `(T, B, C, *N)` and the latter
  produces a trajectory of shape `(B, T, C, *N)` (i.e., the batch `B` and time
  `T` axes are swapped).

### There are no custom grid or state classes (C)

* Lean design that only focuses on JAX Arrays and PyTrees allows for tigher
  integration with other libraries in the JAX ecosystem.

### There is no `jax.jit` being used in the package (C)

* `jit` is supposed to be user-facing functionality

### There are only limited shipped visualization routines (C)

* Keeps the package lean and focused on the core functionality.
* Visualization is very personal and problem-specific.
