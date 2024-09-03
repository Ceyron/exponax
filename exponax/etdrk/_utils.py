from typing import TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Complex

M = TypeVar("M")


def roots_of_unity(M: int) -> Complex[Array, "M"]:
    """
    Return (complex-valued) array with M roots of unity. Useful to perform
    contour integrals in the complex plane.

    **Arguments:**

    - `M`: The number of roots of unity.

    **Returns:**

    - `roots`: The M roots of unity in an array of shape `(M,)`.
    """
    # return jnp.exp(1j * jnp.pi * (jnp.arange(1, M+1) - 0.5) / M)
    return jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
